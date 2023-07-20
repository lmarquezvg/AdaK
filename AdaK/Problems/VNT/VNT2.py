"""
VNT2 test problem.

R. Viennet, C. Fonteix, and I. Marc, "Multicriteria optimization using a
genetic algorithm for determining a Pareto set," in International Journal of
Systems Science, vol. 27, no. 2, pp. 255-260, 1996.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np

def parameters(m):
    """Returns number of decision variables, lower bounds, and upper bounds"""
    n = 2
    lb = -4*np.ones(n)
    ub = 4*np.ones(n)
    return n, lb, ub

def evaluate(P, m):
    """Evaluates a population for the VNT2 test problem"""
    evaluation = np.zeros((len(P), m))
    evaluation[:,0] = (P[:,0]-2)**2/2+(P[:,1]+1)**2/13+3
    evaluation[:,1] = (P[:,0]+P[:,1]-3)**2/36+(-P[:,0]+P[:,1]+2)**2/8-17
    evaluation[:,2] = (P[:,0]+2*P[:,1]-1)**2/175+(2*P[:,1]-P[:,0])**2/17-13
    return evaluation
