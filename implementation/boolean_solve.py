from polynomial_system import PolynomialSystem
from macaulay_matrix import MacaulayMatrix
import logging
import copy
import sympy as sp

TIMEOUT = 100 # If proper matrix/sol cant be found, an empty is returned
'''
Perform BooleanSolve attack on MQ system.
Input:
    poly_system - MQ system that we will attempt to break

'''
def boolean_solve(poly_system: PolynomialSystem, k: int) -> dict:
    m: int = len(poly_system.equations)
    n: int = len(poly_system.variables)

    logging.info(f'Calculating witness degree for (m,n,k) = {(m, n, k)}')
    witness_degree: int = MacaulayMatrix.calculate_witness_degree(m, n, k)
    logging.debug(f'Calculated witness degree = {witness_degree}')

    counter = 0
    while counter < TIMEOUT:
        k_variables = poly_system.get_random_variables(k)
        logging.debug(f'Chosen k = {k} variables: {k_variables}')
        remain_variables = poly_system.variables - set(k_variables)

        logging.debug(f'Creating adjusted polynomial system to {n-k} variables {remain_variables}')
        adjusted_poly_system = poly_system.generate_adjusted_system(remain_variables)
        
        logging.debug('Calculating Macaulay matrix...')
        macaulay = MacaulayMatrix(witness_degree)
        macaulay.create_macaulay_matrix(adjusted_poly_system)

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


def boolean_solve_solver(poly_system: PolynomialSystem, n: int):
    # Strategy has to be devised for this, either take k = n/2 or k = n - 1 for each booleanSolve
    if n <= 1:
        return
    
    solutions = {}
    k = n // 2

    while len(solutions) < n:
        print(f'Calculations for k={k}')
        bool_sol: dict = boolean_solve(poly_system, k)
        if len(bool_sol) == 0: #No solution for given k -> try another one
            k+=1
            continue

        print(bool_sol)
        solutions.update(bool_sol)
        poly_system.replace_variables(bool_sol)
        print(poly_system.equations)
        k = k//2

    print(solutions)



