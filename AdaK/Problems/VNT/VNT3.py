"""
VNT3 test problem.

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
    lb = -3*np.ones(n)
    ub = 3*np.ones(n)
    return n, lb, ub

def evaluate(P, m):
    """Evaluates a population for the VNT3 test problem"""
    evaluation = np.zeros((len(P), m))
    temp = P[:,0]**2+P[:,1]**2
    evaluation[:,0] = 0.5*temp+np.sin(temp)
    evaluation[:,1] = (3*P[:,0]-2*P[:,1]+4)**2/8+(P[:,0]-P[:,1]+1)**2/27+15
    evaluation[:,2] = 1/(temp+1)-1.1*np.exp(-temp)
    return evaluation
