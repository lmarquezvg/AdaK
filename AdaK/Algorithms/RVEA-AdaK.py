"""
RVEA-AdaK.
"""

import numpy as np
from scipy.spatial import distance

from Public.Population import population
from Public.UploadTestProblem import uploadTestProblem
from Public.UniformPoints import uniformPoints
from Public.RandomPopulation import randomPopulation
from Public.GenerateOffspring import generateOffspring
from Public.EfficientNonDominatedSort import efficientNonDominatedSort
from Public.NichingPairPotentialAdaptationMethod import nichingPairPotentialAdaptationMethod

def main(N, problem, m, max_generations, flag):
    """Runs main framework of RVEA-AdaK"""
    parameters, evaluate = uploadTestProblem(problem)
    n, lb, ub = parameters(m)
    W = uniformPoints(N, m)
    N = len(W)
    pc, nc = 1, 30
    pm, nm = 1/n, 20
    alpha = 2
    fr = 0.1
    P = randomPopulation(N, n, m, lb, ub, evaluate)
    zmin = np.min(P.obj, axis=0)
    V = np.copy(W)
    generations = 0
    
    fstep = 0.05
    flast = 0.9
    A = np.copy(P.obj)
    Wadapt = np.copy(W)
    r = np.ones(m)
    
    Data_gen = []
    
    while generations < max_generations:
        M = matingSelection(P, N)
        Q = generateOffspring(M, N, m, lb, ub, pc, nc, pm, nm, evaluate)
        R = population(np.vstack((P.dec, Q.dec)), np.vstack((P.obj, Q.obj)))
        zmin = np.min(np.vstack((zmin, R.obj)), axis=0)
        generations += 1
        P = environmentalSelection(R, V, zmin, (generations/max_generations)**alpha)
        
        if np.floor(generations % (fr*max_generations)) == 0:
            r = referenceVectorAdaptation(P, zmin)
        
        A, Wadapt = nichingPairPotentialAdaptationMethod(A, Q, P, Wadapt, W, N, zmin, fstep, flast, generations, max_generations, flag)[:2]
        V = Wadapt*r
        
        Data_gen.append(P.obj)
        
    return P, Data_gen

def matingSelection(P, N):
    """Selects random parent population"""
    O = N+1 if N%2 == 1 else N
    if len(P.dec) > 1:
        indexes = np.array([np.random.choice(len(P.dec), 2, replace=False) for i in range(0, O//2)])
    else:
        indexes = np.zeros((O//2, 2), dtype=int)
    M = P.dec[np.hstack((indexes[:,0], indexes[:,1]))]
    return M

def environmentalSelection(R, V, zmin, theta):
    """Returns population with best individuals"""
    NV, m = np.shape(V)
    Rprime = R.obj-zmin
    cosine = 1-distance.cdist(V, V, 'cosine')
    np.fill_diagonal(cosine, 0)
    gamma = np.min(np.arccos(cosine), axis=1)
    gamma[gamma == 0] = 1e-12
    Angle = np.arccos(1-distance.cdist(Rprime, V, 'cosine'))
    associate = np.argmin(Angle, axis=1)
    Next = -np.ones(NV, dtype=int)
    for i in np.unique(associate):
        current = np.where(associate == i)[0]
        APD = (1+m*theta*Angle[current,i]/gamma[i])*np.linalg.norm(Rprime[current,:], axis=1)
        best = np.argmin(APD)
        Next[i] = current[best]
    P = population(R.dec[Next[Next!=-1]], R.obj[Next[Next!=-1]])
    return P

def referenceVectorAdaptation(P, zmin):
    """Returns the ranges of each objective function"""
    Fronts = efficientNonDominatedSort(P.obj)
    zmax = np.max(P.obj[Fronts[0]], axis=0)
    return zmax-zmin
