import numpy as np
import sympy as sp
import itertools
import pennylane as qml

from sympy import grevlex, ZZ
from sortedcollections import OrderedSet


def generate_monomials(symbols: set, d: int, domain = ZZ) -> set:
    '''
    Generate all monomials up to given degree. Monomials are sorted from lowest to highest degree. Additionally,
    lexicographical order is preserved between generated monomials.
    Input:
        symbols - List of symbols to be used in monomials
        d - Degree to up which monomials should be generated

    Output:
        List of all monomials (sympy.Poly) up to degree of d
    '''
    monomials = OrderedSet()
    
    if d < 0:
        raise ValueError('Degree has to be a non negative value')
    
    monomials.add(sp.Poly(1, list(symbols))) # Case for zero degree monomial
    for degree in range(1, d+1):
        available_monomials = [p for p in itertools.product(symbols, repeat=degree)]
        for combination in available_monomials:
            new_monomial = 1
            for symbol in combination:
                new_monomial = new_monomial * symbol
            monomials.add(sp.Poly(new_monomial, list(symbols), domain=domain))

    return monomials


def create_monomial_tuples(monomials: set[sp.Poly]) -> list[tuple]:
    '''
    Create tuples for each monomial representing the degree of each symbol present in expression. As input we should
    pass only monomials as only the first terms is taken into consideration when creating tuple.

    Input:
        monomials - Set of monomials where each monomial is an sympy.Poly object
    Output:
        List of tuples where each tuple represents the degree of all variables inside single expression
    '''
    poly_tupples = []
    for m in monomials:
        mono_tuple = list(m.as_dict().keys())[0]
        poly_tupples.append(mono_tuple)
    return poly_tupples


def poly_to_vector(poly: sp.Poly, monomials: set[sp.Poly]) -> list:
    '''
    Transform a polynomial into a vector representation based on provided list of monomials. Vector values are based
    on coefficients of respective terms in polynomial.
    Input:
        poly - Polynomial (sympy.Poly) for which we create a vector representation.
        monomials - Set of monomials (sympy.Poly) that are used to determine vector values
    Output:
        Numpy array representing coefficients of polynomial related to monomials.
    '''
    monomial_tuples = create_monomial_tuples(monomials)
    poly_vector: list = [0]*len(monomial_tuples)
    terms: dict = poly.as_dict()
    
    for degree_tuple, coef in terms.items():
        if degree_tuple in monomial_tuples:
            idx = monomial_tuples.index(degree_tuple)
            poly_vector[idx] = coef
    
    return poly_vector


def create_grevlex_monomials(var_count: int) -> list:
    '''
    Generate monomials ordered in reverse graded lexographical order (maybe we can adjust first function later)
    Input:
        var_count: int - Number of variables
    Output:
        List of terms sorted in 'grevlex' order. Can be used to create polynomials.
    '''
    symbols = [sp.Symbol(f'x_{i+1}') for i in range(var_count)]
    monomials = generate_monomials(symbols, 2) 
    
    poly: sp.Poly = sum(list(monomials))
    return poly.as_expr().as_ordered_terms(order=grevlex)


def gen_possible_coefs(k: int) -> list[tuple]:
    coefs = [0, 1]
    return list(itertools.product(coefs, repeat=k))


def closest_2_power(n: int) -> int:
    '''
    Return closest power of 2 that is greater or equal than input number
    Input:
        n - Integer
    Return:
        m = 2^k where m >= n and k: int
    '''
    if n == 0:
        return 0
    else:
        return 1<<(n-1).bit_length()


def pad_array(arr: np.array) -> np.array:
    '''
    Pads a 2D numpy array into a matrix of N x N where N is the closest power of 2 in comparison
    to the original size of array (Matrix is padded with zeroes)
    Input:
        arr - Original 2D array
    Return:
        Padded matrix of size N X N 
    '''
    n, m = arr.shape[0], arr.shape[1]
    max_size = closest_2_power(max(n, m))
    result = np.zeros((max_size, max_size))
    result[:n,:m] = arr
    for i in range(n, max_size):
        result[i] = result[i - n]
    return result


def matrix_into_unitaries(arr: np.array):
    '''
    Transform input matrix of shape N x N into a linear combination of unitary matrices.
    
    Input: arr - Numpy array of shape n x n (It has to be a square matrix!)
    
    Output: coefs, unitaries - Tuple which represents a linear combination of unitary matrices that is also an input matrix
    '''
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input matrix has to be a square matrix")
    else:
        LCU = qml.pauli_decompose(arr)
        return LCU.coeffs, LCU.ops
    

def make_symmetric(arr: np.array):
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[0]):
            if (j < i):
                arr[i][j] = arr[j][i] = (arr[i][j] +
                                         arr[j][i]) // 2


def get_classic_probabilities(A: np.array, b: np.array):
        if np.linalg.det(A) == 0: #Singular matrix check
            x, res, rank, s = np.linalg.lstsq(A , b, rcond=None)
            # print(f'Residuals: {res}')
            # print(f'Singular vals: {s}
            print(f'Norm: {np.linalg.norm(x)}')
            print(f'Is consistent: {np.allclose(A @ x, b)}')
            if(np.linalg.norm(x) != 0):
                print(f'Classic probabilities: {(x / np.linalg.norm(x)) ** 2}')
        else:
            print('NOT SINGULAR!')
            A_inv = np.linalg.inv(A)
            x = np.dot(A_inv, b)
            print(f'Classic probabilities: {(x / np.linalg.norm(x)) ** 2}')
            return (x / np.linalg.norm(x)) ** 2