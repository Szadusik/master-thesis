import numpy as np
import sympy as sp
from sortedcollections import OrderedSet
import random

''' Class to store polynomial equations and information about them'''

class PolynomialSystem:
    def __init__(self) -> None:
        self.equations: list = []
        self.variables: OrderedSet = OrderedSet()

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
        self.assign_gens_to_equations(variables)

    '''
    Apply/change generating symbols of polynomials inside the system.
    Input:
        variables - List of symbols defining new system of variables
    '''
    def assign_gens_to_equations(self, variables: list | set) -> None:
        self.variables = OrderedSet(variables)
        for equation in self.equations:
            equation = sp.Poly(equation, list(self.variables))

    '''
    Get k random variables from a system.
    Input:
        k - number of variables to retrieve
    '''
    def get_random_variables(self, k: int) -> list[sp.Symbol]:
        return random.sample(self.variables, k)



system = PolynomialSystem()
system.load_equations_from_file("/Users/lukaszsochacki/Desktop/Studia/Magisterka/master-thesis/implementation/test_data/test.txt")