from polynomial_system import PolynomialSystem
from boolean_solve import boolean_solve
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                    filename='myapp.log', 
                    level=logging.DEBUG)
system = PolynomialSystem()
system.load_equations_from_file("/Users/lukaszsochacki/Desktop/Studia/Magisterka/master-thesis/implementation/test_data/test2.txt")

boolean_solve(system, 2)