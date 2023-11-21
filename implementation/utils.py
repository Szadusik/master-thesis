import numpy as np
import sympy as sp
import itertools
from sortedcollections import OrderedSet

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


'''
Create tuples for each monomial representing the degree of each symbol present in expression. As input we should
pass only monomials as only the first terms is taken into consideration when creating tuple.

Input:
    monomials - Set of monomials where each monomial is an sympy.Poly object
Output:
    List of tuples where each tuple represents the degree of all variables inside single expression
'''
def create_monomial_tuples(monomials: set[sp.Poly]) -> list[tuple]:
    poly_tupples = []
    for m in monomials:
        mono_tuple = list(m.as_dict().keys())[0]
        poly_tupples.append(mono_tuple)
    return poly_tupples


'''
Transform a polynomial into a vector representation based on provided list of monomials. Vector values are based
on coefficients of respective terms in polynomial.
Input:
    poly - Polynomial (sympy.Poly) for which we create a vector representation.
    monomials - Set of monomials (sympy.Poly) that are used to determine vector values
Output:
    Numpy array representing coefficients of polynomial related to monomials.
'''
def poly_to_vector(poly: sp.Poly, monomials: set[sp.Poly]) -> np.ndarray:
    monomial_tuples = create_monomial_tuples(monomials)
    poly_vector: np.ndarray = np.zeros(len(monomial_tuples))
    terms: dict = poly.as_dict()
    
    for degree_tuple, coef in terms.items():
        if degree_tuple in monomial_tuples:
            idx = monomial_tuples.index(degree_tuple)
            poly_vector[idx] = coef
    
    return poly_vector
