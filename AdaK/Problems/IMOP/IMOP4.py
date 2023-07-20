"""
IMOP4 test problem.

Y. Tian, R. Cheng, X. Zhang, M. Li, and Y. Jin, "Diversity Assessment of 
Multi-Objective Evolutionary Algorithms: Performance Metric and Benchmark 
Problems [Research Frontier]," in IEEE Computational Intelligence Magazine, 
vol. 14, no. 3, pp. 61–74, 2019.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np

def parameters(m):
    """Returns number of decision variables, lower bounds, and upper bounds"""
    K = 5
    L = 5
    n = K+L
    lb = np.zeros(n)
    ub = np.ones(n)
    return n, lb, ub

def evaluate(P, m):
    """Evaluates a population for the IMOP4 test problem"""
    K = 5
    a1 = 0.05
    evaluation = np.zeros((len(P), m))
    y1 = np.mean(P[:,:K], axis=1)**a1
    g = np.sum((P[:,K:]-0.5)**2, axis=1)
    evaluation[:,0] = (1+g)*y1
    evaluation[:,1] = (1+g)*(y1+np.sin(10*np.pi*y1)/10)
    evaluation[:,2] = (1+g)*(1-y1)
    return evaluation
