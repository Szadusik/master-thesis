from polynomial_system import PolynomialSystem
from boolean_solve import boolean_solve
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                    filename='myapp.log', 
                    level=logging.DEBUG)
system = PolynomialSystem()
system.load_equations_from_file('implementation/test_data/mq-chall-test.txt')
results = boolean_solve(system, 6)
system.verify_solutions(results)