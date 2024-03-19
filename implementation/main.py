import logging
'''
Classic version
'''
# from classic.polynomial_system import PolynomialSystem
# from classic.boolean_solve import boolean_solve
# import logging

# logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
#                     filename='myapp.log', 
#                     level=logging.DEBUG)
# system = PolynomialSystem()
# system.load_equations_from_file('implementation/test_data/mq-chall-test.txt')
# results = boolean_solve(system, 9)
# system.verify_solutions(results)
'''
VQLS version
'''
from quantum.vqls_boolean_solve import boolean_solve
from quantum.polynomial_system import PolynomialSystem

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                    filename='myapp.log', 
                    level=logging.DEBUG)
system = PolynomialSystem()
system.load_equations_from_file('implementation/test_data/mq-chall-test.txt')
results = boolean_solve(system, 9)
system.verify_solutions(results)

