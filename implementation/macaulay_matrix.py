import numpy as np
import sympy as sp
import itertools


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
Generate all monomials up to given degree. 
Input:
    symbols - List of symbols to be used in monomials
    d - Degree to up which monomials should be generated

Output:
    List of all monomials (sympy.Poly) up to degree of d
'''

def generate_monomials(symbols: set, d: int) -> set:
    monomials = set()
    
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

# print(calculate_witness_degree(5,5,2))

x, y, z = sp.symbols('x y z')
symbols = set([x, y, z])
monomials = generate_monomials(symbols, 2)
print(monomials)