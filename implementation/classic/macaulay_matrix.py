import numpy as np
import sympy as sp
from polynomial_system import PolynomialSystem
from utils import generate_monomials, poly_to_vector
from sympy.solvers.solveset import linsolve

class MacaulayMatrix:
    def __init__(self, D: int) -> None:
        self.D: int= D
        self.matrix: list[list] = []

    ''' 
    Calculate witness degree which used to determine degree of Macaulay matrix. Calculation based on Hilbert series. 
    Index of non- element is counted from lowest degree element.

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

        q, r = sp.div(nominator, denominator, gens=[t])
        poly_res: sp.Poly = (q + r).as_poly()
        
        coeffs = poly_res.all_coeffs()
        coeffs.reverse()
        print(coeffs)
        nonzero_indices = np.nonzero(coeffs)[0] #index of first non-zero element
        return nonzero_indices[0] + 1 if len(nonzero_indices) != 0 else len(coeffs)


    @staticmethod
    def calculate_witness_degree_alternative(m: int, n: int, k: int) -> int:
        t = sp.Symbol('t')
        nominator = (1 + t)**(n - k)
        denominator = (1 - t)*((1 + t**2)**m)

        q, r = sp.div(nominator, denominator, gens=[t])
        poly_res: sp.Poly = (q + r).as_poly()
        
        print(poly_res)
        coeffs = poly_res.all_coeffs()
        coeffs.reverse()
        for i in range(len(coeffs)):
            if coeffs[i] <=0:
                return i
        return len(coeffs)

    
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
        for mono in monomials: 
            d_mono = mono.total_degree() 
            for f in poly_system.equations:
                d_poly = f.total_degree()
                #print((d_mono, d_poly))
                if d_mono <= (self.D - d_poly): #Which polynomials should be added to Macaulay matrix as rows
                    rows_polynomials.append(sp.simplify(f) * mono)
        
        zero_vector = [0] * len(monomials)
        macaulay_matrix : list[list] = [] # Creating macaulay matrix
        for p in rows_polynomials:
            vector = poly_to_vector(p, monomials)
            if vector == zero_vector or vector in macaulay_matrix:
                continue
            macaulay_matrix.append(vector)

        self.matrix = macaulay_matrix


    def solve_macaulay_equation(self) -> bool:
        A = np.array(self.matrix, dtype=float)
        if not np.any(A): # Have to be handled outside function
            return False
        
        r = np.zeros(A.shape[0])
        r[-1] = 1

        solution, residuals, rank, singular_values = np.linalg.lstsq(A.astype('float'), r.astype('float'), rcond=None)

        is_consistent = np.allclose(A @ solution, r)
        return not is_consistent
    

    def get_matrix_size(self) -> tuple[int, int]:
        n = len(self.matrix)
        m = 0
        if n > 0:
            m = len(self.matrix[0])
        return n,m
        