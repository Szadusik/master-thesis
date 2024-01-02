import numpy as np
import sympy as sp
from sympy import ZZ
from sortedcollections import OrderedSet
import random

''' Class to store polynomial equations and information about them'''

class PolynomialSystem:
    def __init__(self) -> None:
        self.equations: list[sp.Poly] = []
        self.variables: OrderedSet = OrderedSet()
        self.domain = ZZ

    '''
    Load polynomial equations from given file. Each equation is placed in one line of file and their format should be
    valid with expression format of Symp. Additionally, equation has to be written as polynomial.
    Input:
        filepath - path to file containing equations that will be read
    
    Output:
        None - Equations/variables will be stored inside class
    '''
    def load_equations_from_file(self, filepath: str) -> None:
        file = open(filepath, "r+")
        variables: list = []
        for equation in file:
            poly: sp.Poly = sp.polys.polytools.poly_from_expr(equation)
            self.equations.append(poly[0])
            for variable in poly[1]['gens']:
                variables.append(variable)

        variables.sort(key=str)
        self.variables = OrderedSet(variables)
        for equation in self.equations:
            equation = sp.Poly.from_poly(equation, gens=equation.gens, domain=self.domain)


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
    
    
    def replace_variables(self, var_map: dict):
        for var in var_map.keys():
            if var in self.variables:
                self.variables.remove(var)
        
        for eq in self.equations:
            idx = self.equations.index(eq)
            reduced_eq = eq.subs(var_map)
            self.equations[idx] = sp.Poly(reduced_eq)
