"""
DTLZ2_MINUS test problem.

K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, "Scalable Test Problems for 
Evolutionary Multiobjective Optimization," in Evolutionary Multiobjective 
Optimization. Theoretical Advances and Applications, pp. 105-145, 2005.

H. Ishibuchi, Y. Setoguchi, H. Masuda, and Y. Nojima, "Performance of 
Decomposition-Based Many-Objective Algorithms Strongly Depends on Pareto Front 
Shapes," in IEEE Transactions on Evolutionary Computation, vol. 21, no. 2, 
pp. 169-190, 2017.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np

def parameters(m):
    """Returns number of decision variables, lower bounds, and upper bounds"""
    k = 10
    n = m-1+k
    lb = np.zeros(n)
    ub = np.ones(n)
    return n, lb, ub

def evaluate(P, m):
    """Evaluates a population for the DTLZ2_MINUS test problem"""
    N, n = np.shape(P)
    g = np.sum((P[:,m-1:]-0.5)**2, axis=1)[:,np.newaxis]
    return -np.tile(1+g, (1, m))*np.fliplr(np.cumprod(np.hstack((np.ones((N, 1)), np.cos(P[:,:m-1]*np.pi/2))), axis=1))*np.hstack((np.ones((N, 1)), np.sin(P[:,m-2::-1]*np.pi/2)))
