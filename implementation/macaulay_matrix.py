import numpy as np
import sympy as sp
from polynomial_system import PolynomialSystem
from utils import generate_monomials, poly_to_vector
import logging
from sympy.solvers.solveset import linsolve

class MacaulayMatrix:
    def __init__(self, D: int) -> None:
        self.D: int= D
        self.matrix: list[list] = []

    ''' 
    Calculate witness degree which used to determine degree of Macaulay matrix. Calculation based on Hilbert series. 
    Index of non-zero element is counted from lowest degree element.

    Input:
        m - Number of equations inside system
        n - Number of variables present in the system
        k - Size of subset of variables (k <= n)
    '''
    @staticmethod
    def calculate_witness_degree(m: int, n: int, k: int) -> int:
        t = sp.Symbol('t')
        nominator = (1 + t)**(n - k)
        denominator = (1 - t)*((1 + t**2)**m)

        q, r = sp.div(nominator, denominator)
        poly_res: sp.Poly = (q + r).as_poly()
        
        coeffs = poly_res.all_coeffs()
        coeffs.reverse()
        nonzero_indices = np.nonzero(coeffs)[0]
        return nonzero_indices[0] if len(nonzero_indices) != 0 else -1


    ''' 
    Generate Macaulay matrix for given system of polynomial equations and predefinied degree of matrix.
    Input:
        poly_system - System of polynomial equations

    Output:
        Macaulay matrix of degree D based on poly_system.
    '''
    def create_macaulay_matrix(self, poly_system: PolynomialSystem):
        system_variables: set[sp.Symbol] = poly_system.variables
        monomials: set[sp.Poly] = generate_monomials(system_variables, self.D)

        rows_polynomials = []

        #print(f'Poly system: {poly_system.equations}')
        for mono in monomials:
            d_mono = mono.total_degree()
            for f in poly_system.equations:
                d_poly = f.total_degree()
                #print((d_mono, d_poly))
                if d_mono <= (self.D - d_poly):
                    rows_polynomials.append(sp.simplify(f) * mono)
        
        zero_vector = [0] * len(monomials)
        macaulay_matrix : list[list] = []
        for p in rows_polynomials:
            vector = poly_to_vector(p, monomials)
            if vector == zero_vector or vector in macaulay_matrix:
                continue
            macaulay_matrix.append(vector)

        self.matrix = macaulay_matrix


    def solve_macaulay_equation(self) -> bool:
        if len(self.matrix) == 0: # Have to be handled outside function
            return False
        
        symbols = [sp.Symbol(f'u_{i}') for i in range(len(self.matrix[0]))] # Asssume len(matrix) > 0
        r = [0] * len(self.matrix)
        r[-1] = 1

        system = []
        for eq in self.matrix: # Each row -> coef of vector u
            new_eq = 0
            for i in range(len(symbols)):
                new_eq += (symbols[i] * eq[i]) + r[self.matrix.index(eq)]
            system.append(new_eq)

        # solution = linsolve(system, symbols) - Maybe can be used
        return len(sp.solve(system, symbols)) == 0

        