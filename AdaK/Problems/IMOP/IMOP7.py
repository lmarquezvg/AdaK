"""
IMOP7 test problem.

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
    """Evaluates a population for the IMOP7 test problem"""
    K = 5
    a2 = 0.05
    a3 = 10
    evaluation = np.zeros((len(P), m))
    y2 = np.mean(P[:,:K:2], axis=1)**a2
    y3 = np.mean(P[:,1:K:2], axis=1)**a3
    g = np.sum((P[:,K:]-0.5)**2, axis=1)
    evaluation[:,0] = (1+g)*np.cos(np.pi/2*y2)*np.cos(np.pi/2*y3)
    evaluation[:,1] = (1+g)*np.cos(np.pi/2*y2)*np.sin(np.pi/2*y3)
    evaluation[:,2] = (1+g)*np.sin(np.pi/2*y2)
    r = np.minimum(np.minimum(np.abs(evaluation[:,0]-evaluation[:,1]), np.abs(evaluation[:,1]-evaluation[:,2])), np.abs(evaluation[:,2]-evaluation[:,0]))
    return evaluation+np.tile(10*np.maximum(0, r[:,np.newaxis]-0.1), (1, 3))
