import logging
import copy
import sympy as sp
import numpy as np

from quantum.polynomial_system import PolynomialSystem
from quantum.macaulay_matrix import MacaulayMatrix
from quantum.utils import gen_possible_coefs, get_classic_probabilities
from quantum.vqls import VQLS


def boolean_solve(poly_system: PolynomialSystem, k: int) -> dict:
    '''
    Perform BooleanSolve attack on MQ system.
    Input:
        poly_system - MQ system that we will attempt to break
    Output:
        Set of solutions for provided polynomial system

    '''
    m: int = len(poly_system.equations)
    n: int = len(poly_system.variables)

    logging.info(f'Calculating witness degree for (m,n,k) = {(m, n, k)}')
    witness_degree: int = MacaulayMatrix.calculate_witness_degree_alternative(m, n, k)
    logging.debug(f'Calculated witness degree = {witness_degree}')

    solutions = []
    mac_min, mac_max = 10**6, 0
    symbols_combinations = poly_system.get_k_variables(k)
    for symbols in symbols_combinations:
        coefs_perm = gen_possible_coefs(k)
        print(f'Symbols: {symbols}')
        for coefs in coefs_perm:
            logging.debug(f'Chosen k = {k} variables {symbols} with coefs {coefs}')
            val_map = dict(zip(symbols, coefs))

            new_variables = poly_system.variables - set(symbols)
            logging.debug(f'Creating adjusted polynomial system with {n-k} variables {new_variables}')
            adjusted_poly_system = copy.deepcopy(poly_system)
            adjusted_poly_system.specialize_variables(val_map)

            logging.debug('Calculating Macaulay matrix...')
            macaulay = MacaulayMatrix(witness_degree)
            macaulay.create_macaulay_matrix(adjusted_poly_system)
            if macaulay.solve_macaulay_equation():
                call_VQLS(macaulay)

                mac_size = macaulay.get_matrix_size()
                mac_min, mac_max = min(mac_min, mac_size[0]), max(mac_max, mac_size[0])
                #print(f'Potential solution for {val_map}')
                #print(f'Macaulay matrix size: {macaulay.get_matrix_size()}')
                nk_solution = adjusted_poly_system.solve_equation_system()
                logging.debug(f'Found solutions for {n-k} variables: {nk_solution}')
                if(len(nk_solution) != 0):
                    # print(nk_solution)
                    # print(nk_solution[0])
                    # print(adjusted_poly_system.variables)
                    #print(f'Is system linear: {adjusted_poly_system.is_system_linear()}')
                    solution_map = val_map
                    solution_map.update(dict(zip(list(adjusted_poly_system.variables), list(nk_solution[0]))))
                    if solution_map not in solutions:
                        solutions.append(solution_map)
    print(solutions)
    print(f'Macaulay rows bounds: {(mac_min, mac_max)}')
    return solutions


def call_VQLS(macaulay: MacaulayMatrix):
    A = np.array(macaulay.matrix)
    b = np.zeros(A.shape[0])
    b[A.shape[0]-1] = 1

    A_override = np.array([[1.0,  0.0,  0.0,  0.0,  0.4, 0.0,  0.0,  0.0], 
                [0.0,  1.0,  0.0,  0.0,  0.0,  0.4, 0.0,  0.0],
                [0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
                [0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0 ],
                [0.4, 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0 ],
                [0.0,  0.4, 0.0,  0.0,  0.0,  1.0,  0.0,  0.0 ],
                [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0, 0.0 ],
                [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0 ]])
    b_override = np.array([0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339,0.35355339, 0.35355339])
    #VQLS.solve_linear_equation(A, b)
    VQLS.solve_linear_equation(A_override, b_override)