"""
MOEA/D.

Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
Decomposition," in IEEE Transactions on Evolutionary Computation, vol. 11, 
no. 6, pp. 712-731, 2007.

H. Li and Q. Zhang, "Multiobjective Optimization Problems With Complicated 
Pareto Sets, MOEA/D and NSGA-II," in IEEE Transactions on Evolutionary 
Computation, vol. 13, no. 2, pp. 284-302, 2009.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np
from scipy.spatial import distance

from Public.UploadTestProblem import uploadTestProblem
from Public.UniformPoints import uniformPoints
from Public.RandomPopulation import randomPopulation
from Public.GenerateOffspring import generateOffspring
from Public.EfficientNonDominatedSort import efficientNonDominatedSort

def main(N, problem, m, max_generations):
    """Runs main framework of MOEA/D"""
    parameters, evaluate = uploadTestProblem(problem)
    n, lb, ub = parameters(m)
    W = uniformPoints(N, m)
    N = len(W)
    pc, nc = 1, 20
    pm, nm = 1/n, 20
    T = int(np.ceil(N/10))
    delta = 0.9
    nr = int(np.ceil(N/100))
    B = np.argsort(distance.cdist(W, W, 'euclidean'), axis=1)[:,:T]
    P = randomPopulation(N, n, m, lb, ub, evaluate)
    zmin = np.min(P.obj, axis=0)
    generations = 0
    
    fstep = 0.05
    flast = 0.9
    Wrange = np.copy(W)
    
    Data_gen = []
    
    while generations < max_generations:
        for i in range(0, N):
            if np.random.rand() < delta:
                I = np.random.permutation(B[i])
            else:
                I = np.random.permutation(N)
            M = P.dec[I[:2]]
            y = generateOffspring(M, 1, m, lb, ub, pc, nc, pm, nm, evaluate)
            zmin = np.minimum(zmin, y.obj)
            offspring = ASF(y.obj, Wrange[I], zmin)
            neighbors = ASF(P.obj[I], Wrange[I], zmin)
            replace = I[np.where(offspring <= neighbors)[0][:nr]]
            P.dec[replace] = y.dec
            P.obj[replace] = y.obj
        generations += 1
        
        if np.floor(generations % (fstep*max_generations)) == 0 and generations <= flast*max_generations:
            Wrange = adaptRanges(P, W, zmin)
        
        Data_gen.append(np.copy(P.obj))
        
    return P, Data_gen

def ASF(P, W, zmin):
    """Evaluates a population using the achievement scalarizing function"""
    Weights = np.copy(W)
    Weights[Weights < 1e-3] = 1e-3
    return np.max(np.abs(P-zmin)/Weights, axis=1)

def adaptRanges(P, W, zmin):
    """Adapts the original reference set to the ranges of each objective function"""
    Fronts = efficientNonDominatedSort(P.obj)
    zmax = np.max(P.obj[Fronts[0]], axis=0)
    Wscale = W*(zmax-zmin)
    denom = np.sum(Wscale, axis=1)
    denom[denom == 0] = 1e-12
    return Wscale/denom[:,np.newaxis]
