import numpy as np
import sympy as sp
import itertools
from polynomial_system import PolynomialSystem
from sortedcollections import OrderedSet

'''
Calculate witness degree which used to determine degree of Macaulay matrix. Calculation based on Hilbert series. 
Index of non-zero element is counted from lowest degree element.

Input:
    m - Number of equations inside system
    n - Number of variables present in the system
    k - Size of subset of variables (k <= n)
'''

def calculate_witness_degree(m: int, n: int, k: int) -> int:
    t = sp.Symbol('t')
    nominator = (1 + t)**(n - k)
    denominator = (1 - t)*((1 + t**2)**m)

    q, r = sp.div(nominator, denominator)
    poly_res: sp.Poly = (q + r).as_poly()
    
    coeffs = poly_res.all_coeffs()
    coeffs.reverse()
    nonzero_indices = np.nonzero(coeffs)[0]
    return nonzero_indices[0] + 1 if len(nonzero_indices) != 0 else 0


'''
Generate all monomials up to given degree. Monomials are sorted from lowest to highest degree. Additionally,
lexicographical order is preserved between generated monomials.
Input:
    symbols - List of symbols to be used in monomials
    d - Degree to up which monomials should be generated

Output:
    List of all monomials (sympy.Poly) up to degree of d
'''

def generate_monomials(symbols: set, d: int) -> set:
    monomials = OrderedSet()
    
    if d < 0:
        raise ValueError('Degree has to be a non negative value')
    
    zero_degree_poly = sp.Poly(1, list(symbols))
    monomials.add(zero_degree_poly) # Case for zero degree monomial
    
    for degree in range(1, d+1):
        available_monomials = [p for p in itertools.product(symbols, repeat=degree)]
        for combination in available_monomials:
            new_monomial = 1
            for symbol in combination:
                new_monomial = new_monomial * symbol
            monomials.add(sp.Poly(new_monomial, list(symbols)))

    return monomials


def create_monomial_tuples(monomials: set[sp.Poly]) -> list[tuple]:
    poly_tupples = []
    for m in monomials:
        mono_tuple = list(m.as_dict().keys())[0]
        poly_tupples.append(mono_tuple)
    return poly_tupples
    

def poly_to_vector(poly: sp.Poly, monomial_tuples: list[tuple]) -> np.ndarray:
    poly_vector: np.ndarray = np.zeros(len(monomial_tuples))
    terms: dict = poly.as_dict()
    
    for degree_tuple, coef in terms.items():
        if degree_tuple in monomial_tuples:
            idx = monomial_tuples.index(degree_tuple)
            poly_vector[idx] = coef
    
    return poly_vector


def create_macaulay_matrix(poly_system: PolynomialSystem, degree: int):
    system_variables: set[sp.Symbol] = poly_system.variables
    monomials: set[sp.Poly] = generate_monomials(system_variables, degree)

    rows_polynomials = []
    for mono in monomials:
        d_mono = mono.degree()
        for f in poly_system.equations:
            d_poly = f.degree()
            if d_mono <= degree - d_poly:
                rows_polynomials.append(f * mono)
    
    column_tuples: list[tuple] = create_monomial_tuples(monomials)
    zero_vector = np.zeros(len(column_tuples))

    vectorized_p : list[np.ndarray] = []
    for p in rows_polynomials:
        vector = poly_to_vector(p, column_tuples)
        if np.array_equal(vector, zero_vector):
            continue
        vectorized_p.append(vector)

    macaulay_matrix = np.array(vectorized_p)        
    return np.unique(macaulay_matrix, axis=0)


'''
Generate monomial tuples where each tuple represent a degress of each variable that makes a monomial

'''