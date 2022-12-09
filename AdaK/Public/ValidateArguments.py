"""
Validate arguments.
"""

import sys

from Public.ValidateAlgorithm import validateAlgorithm
from Public.ValidateTestProblem import validateTestProblem

def validateArguments(algorithm, N, problem, m, max_generations, runs):
    """Validates given arguments"""
    if (validateAlgorithm(algorithm) is False):
        sys.exit('Invalid MOEA name. Use: main.py --help')
    if (N <= 0):
        sys.exit('Invalid value for the population size. Use: main.py --help')
    if (validateTestProblem(problem) is False):
        sys.exit('Invalid MOP name. Use: main.py --help')
    if (m < 2):
        sys.exit('Invalid value for the number of objective functions. Use: main.py --help')
    if (max_generations < 0):
        sys.exit('Invalid value for the maximum number of generations. Use: main.py --help')
    if (runs <= 0):
        sys.exit('Invalid value for the number of independent runs. Use: main.py --help')
