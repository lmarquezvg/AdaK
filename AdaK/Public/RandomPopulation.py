"""
Random population.
"""

import numpy as np

from Public.Population import population

def randomPopulation(N, n, m, lb, ub, evaluate):
    """Generates a random population"""
    decision = lb+(np.random.rand(N, n)*(ub-lb))
    objective = evaluate(decision, m)
    return population(decision, objective)
