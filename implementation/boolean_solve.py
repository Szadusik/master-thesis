from polynomial_system import PolynomialSystem
from macaulay_matrix import MacaulayMatrix
from utils import gen_possible_coefs

import logging
import copy
import sympy as sp
import numpy as np

TIMEOUT = 100 # If proper matrix/sol cant be found, an empty is returned
'''
Perform BooleanSolve attack on MQ system.
Input:
    poly_system - MQ system that we will attempt to break

'''
def boolean_solve(poly_system: PolynomialSystem, k: int) -> dict:
    m: int = len(poly_system.equations)
    n: int = len(poly_system.variables)

    # Calculate witness degree
    logging.info(f'Calculating witness degree for (m,n,k) = {(m, n, k)}')
    witness_degree: int = MacaulayMatrix.calculate_witness_degree(m, n, k)
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
                print(adjusted_poly_system.equations)
                nk_solution = adjusted_poly_system.solve_equation_system()
                logging.debug(f'Found solutions for {n-k} variables: {nk_solution}')
                print(nk_solution)


    return

    counter = 0
    while counter < TIMEOUT:
        logging.debug('Solving linear system from Macaulay matrix')
        if macaulay.solve_macaulay_equation():
            logging.debug(f'Linear system is inconsistent, finding solution for {k_variables}...')
            break
    
        logging.debug('Linear system is consistent, retrying new variables...')
        counter+=1

    if(counter == TIMEOUT):
        logging.warn('Timeout has been reached, returning empty solution.')
        return {}
    
    #Finding solutions for n-k variables
    nk_solution = adjusted_poly_system.solve_equation_system()
    logging.debug(f'Found solutions for {n-k} variables: {nk_solution}')
    return nk_solution



