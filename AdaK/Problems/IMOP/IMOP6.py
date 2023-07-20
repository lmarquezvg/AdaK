"""
IMOP6 test problem.

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
    """Evaluates a population for the IMOP6 test problem"""
    K = 5
    a2 = 0.05
    a3 = 10
    evaluation = np.zeros((len(P), m))
    y2 = np.mean(P[:,:K:2], axis=1)**a2
    y3 = np.mean(P[:,1:K:2], axis=1)**a3
    g = np.sum((P[:,K:]-0.5)**2, axis=1)
    r = np.maximum(0, np.minimum(np.sin(3*np.pi*y2)**2, np.sin(3*np.pi*y3)**2)-0.05)
    evaluation[:,0] = (1+g)*y2+np.ceil(r)
    evaluation[:,1] = (1+g)*y3+np.ceil(r)
    evaluation[:,2] = (0.5+g)*(2-y2-y3)+np.ceil(r)
    return evaluation
