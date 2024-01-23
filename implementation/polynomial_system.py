import random
import re
import itertools
import numpy as np
import sympy as sp

from sympy import ZZ, grevlex
from sortedcollections import OrderedSet
from utils import generate_monomials


''' Class to store polynomial equations and information about them'''
class PolynomialSystem:
    def __init__(self) -> None:
        self.equations: list[sp.Poly] = []
        self.variables: OrderedSet = OrderedSet()
        self.domain = ZZ

    '''
    Load polynomial equations from given file. Format has to be the same as the one present
    in Fumuoka MQ challenge. For now support excluded for GF(2^8)
    Input:
        filepath - path to file containing equations that will be read
    Output:
       Equations/variables will be stored inside class object
    '''
    def load_equations_from_file(self, path: str) -> None:
        mq_file = open(path, "r")
        mq_data = mq_file.readlines()
        info, coefs = mq_data[:5], mq_data[5:]

        #Load file information
        domain = info[0].split(" : ")[1].replace('\n', '')
        n = int(info[1].split(" : ")[1])
        m = int(info[2].split(" : ")[1])

        #Prepare variables and terms (columns in file)
        symbols = [sp.Symbol(f'x_{i+1}') for i in range(n)]
        self.variables = OrderedSet(symbols)
        monomials = generate_monomials(symbols, 2) 
        poly: sp.Poly = sum(list(monomials))
        terms = poly.as_expr().as_ordered_terms(order=grevlex)

        #Read coefficients and apply to terms
        current_coefs = ""
        for line in coefs:
            line = line.replace('\n', '')
            if line == "" or re.match(r'\*+', line):
                continue
            else:
                current_coefs += line
                if ';' in line:
                    current_coefs = current_coefs.replace(' ;', '')
                    poly = self.__create_poly(terms, current_coefs)
                    self.equations.append(poly)
                    current_coefs = ""


    def __create_poly(self, terms, coefs) -> sp.Poly:
        coefs = coefs.split(' ')
        poly = 0
        #print(f'Terms count: {len(terms)}, coefs: {len(coefs)}')
        for term, coef in zip(terms, coefs):
            poly += int(coef)*term
        return sp.Poly(poly)

    '''
    Create a new polynomial system based on the original system. Given new set of variables equations are
    adjusted to new variables. Variables not present in the provided list are considered as constants.
    Input:
        variables - List of symbols defining new system of variables
    '''
    def generate_adjusted_system(self, variables: list | set):
        poly_system = PolynomialSystem()
        poly_system.variables = OrderedSet(variables)
        for equation in self.equations:
            new_equation = sp.Poly.from_poly(equation, *list(variables))
            poly_system.equations.append(new_equation)
        return poly_system
    

    def get_k_variables(self, k: int) -> list[tuple[sp.Symbol]]:
        return list(itertools.combinations(self.variables, k))
    
    '''
    Get k random variables from a system.
    Input:
        k - number of variables to retrieve
    '''
    def get_random_variables(self, k: int) -> list[sp.Symbol]:
        return random.sample(self.variables, k)

    '''
    Solve system equation based on provided variables and equations
    Input:
        None (Parameters are present inside object)
    Return:
        Values for variables that are solution for system
    '''
    def solve_equation_system(self) -> dict:
        return sp.solve(self.equations, self.variables)
    
    
    def specialize_variables(self, var_map: dict):
        for var in var_map.keys():
            if var in self.variables:
                self.variables.remove(var)
        
        for eq in self.equations:
            idx = self.equations.index(eq)
            reduced_eq = eq.subs(var_map)
            self.equations[idx] = sp.Poly(reduced_eq, gens=self.variables)

# system = PolynomialSystem()
# system.load_equations_from_file('implementation/test_data/mq-chall-test.txt')
# print(system.variables)
# print(system.equations)
