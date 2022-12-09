"""
MOEA/D-AdaK.
"""

import numpy as np
from scipy.spatial import distance

from Public.Population import population
from Public.UploadTestProblem import uploadTestProblem
from Public.UniformPoints import uniformPoints
from Public.RandomPopulation import randomPopulation
from Public.GenerateOffspring import generateOffspring
from Public.NichingPairPotentialAdaptationMethod import nichingPairPotentialAdaptationMethod

def main(N, problem, m, max_generations, flag):
    """Runs main framework of MOEA/D-AdaK"""
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
    A = np.copy(P.obj)
    Wadapt = np.copy(W)
    
    Data_gen = []
    
    while generations < max_generations:
        Q = population([], [])
        for i in range(0, N):
            if np.random.rand() < delta:
                I = np.random.permutation(B[i])
            else:
                I = np.random.permutation(N)
            M = P.dec[I[:2]]
            y = generateOffspring(M, 1, m, lb, ub, pc, nc, pm, nm, evaluate)
            Q.dec = np.copy(y.dec) if len(Q.dec) == 0 else np.vstack((Q.dec, y.dec))
            Q.obj = np.copy(y.obj) if len(Q.obj) == 0 else np.vstack((Q.obj, y.obj))
            zmin = np.minimum(zmin, y.obj)
            offspring = ASF(y.obj, Wadapt[I], zmin)
            neighbors = ASF(P.obj[I], Wadapt[I], zmin)
            replace = I[np.where(offspring <= neighbors)[0][:nr]]
            P.dec[replace] = y.dec
            P.obj[replace] = y.obj
        generations += 1
        A, Wadapt, P, B = nichingPairPotentialAdaptationMethod(A, Q, P, Wadapt, W, N, zmin, fstep, flast, generations, max_generations, flag, None, True, True, T, B, ASF)
        
        Data_gen.append(np.copy(P.obj))
        
    return P, Data_gen

def ASF(P, W, zmin):
    """Evaluates a population using the achievement scalarizing function"""
    Weights = np.copy(W)
    Weights[Weights < 1e-3] = 1e-3
    return np.max(np.abs(P-zmin)/Weights, axis=1)
