"""
DTLZ7 test problem.

K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, "Scalable Test Problems for 
Evolutionary Multiobjective Optimization," in Evolutionary Multiobjective 
Optimization. Theoretical Advances and Applications, pp. 105-145, 2005.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np

def parameters(m):
    """Returns number of decision variables, lower bounds, and upper bounds"""
    k = 20
    n = m-1+k
    lb = np.zeros(n)
    ub = np.ones(n)
    return n, lb, ub

def evaluate(P, m):
    """Evaluates a population for the DTLZ7 test problem"""
    N, n = np.shape(P)
    evaluation = np.zeros((N, m))
    g = 1+9*np.mean(P[:,m-1:], axis=1)
    evaluation[:,:m-1] = np.copy(P[:,:m-1])
    evaluation[:,m-1] = (1+g)*(m-np.sum(evaluation[:,:m-1]/np.tile(1+g[:,np.newaxis], (1,m-1))*(1+np.sin(3*np.pi*evaluation[:,:m-1])), axis=1))
    return evaluation
