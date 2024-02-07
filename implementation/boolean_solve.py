from polynomial_system import PolynomialSystem
from macaulay_matrix import MacaulayMatrix
from utils import gen_possible_coefs

import logging
import copy
import sympy as sp
import numpy as np

'''
Perform BooleanSolve attack on MQ system.
Input:
    poly_system - MQ system that we will attempt to break
Output:
    Set of solutions for provided polynomial system

'''
def boolean_solve(poly_system: PolynomialSystem, k: int) -> dict:
    m: int = len(poly_system.equations)
    n: int = len(poly_system.variables)

    # Calculate witness degree
    logging.info(f'Calculating witness degree for (m,n,k) = {(m, n, k)}')
    witness_degree: int = MacaulayMatrix.calculate_witness_degree_alternative(m, n, k)
    logging.debug(f'Calculated witness degree = {witness_degree}')

    solutions = []
    # We will need to look up all combinations of k variables
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
                #print(f'Potential solution for {val_map}')
                nk_solution = adjusted_poly_system.solve_equation_system()
                logging.debug(f'Found solutions for {n-k} variables: {nk_solution}')
                if(len(nk_solution) != 0):
                    # print(nk_solution)
                    # print(nk_solution[0])
                    # print(adjusted_poly_system.variables)
                    solution_map = val_map
                    solution_map.update(dict(zip(list(adjusted_poly_system.variables), list(nk_solution[0]))))
                    if solution_map not in solutions:
                        solutions.append(solution_map)
    print(solutions)
    return solutions



