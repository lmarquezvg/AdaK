"""
RVEA*.

R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, "A Reference Vector Guided 
Evolutionary Algorithm for Many-Objective Optimization," in IEEE Transactions 
on Evolutionary Computation, vol. 20, no. 5, pp. 773-791, 2016.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np
from scipy.spatial import distance

from Public.Population import population
from Public.UploadTestProblem import uploadTestProblem
from Public.UniformPoints import uniformPoints
from Public.RandomPopulation import randomPopulation
from Public.GenerateOffspring import generateOffspring
from Public.EfficientNonDominatedSort import efficientNonDominatedSort

def main(N, problem, m, max_generations):
    """Runs main framework of RVEA*"""
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
    V = np.vstack((W, np.random.rand(N, m)))
    generations = 0
    
    Data_gen = []
    
    while generations < max_generations:
        M = matingSelection(P, N)
        Q = generateOffspring(M, N, m, lb, ub, pc, nc, pm, nm, evaluate)
        R = population(np.vstack((P.dec, Q.dec)), np.vstack((P.obj, Q.obj)))
        zmin = np.min(np.vstack((zmin, R.obj)), axis=0)
        generations += 1
        P = environmentalSelection(R, V, zmin, (generations/max_generations)**alpha)
        
        if np.floor(generations % (fr*max_generations)) == 0:
            V[:N] = referenceVectorAdaptation(P, W, zmin)
        
        V[N:] = referenceVectorRegeneration(P, V[N:], zmin)
        
        Data_gen.append(P.obj)
    
    P = truncation(P, N)
    
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
    Fronts = efficientNonDominatedSort(R.obj)
    R.dec = R.dec[Fronts[0]]
    R.obj = R.obj[Fronts[0]]
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

def referenceVectorAdaptation(P, W, zmin):
    """Adapts the original reference set to the ranges of each objective function"""
    Fronts = efficientNonDominatedSort(P.obj)
    zmax = np.max(P.obj[Fronts[0]], axis=0)
    return W*(zmax-zmin)

def referenceVectorRegeneration(P, V, zmin):
    """Regenerates the additional reference set"""
    NV, m = np.shape(V)
    Pprime = P.obj-zmin
    associate = np.argmax(1-distance.cdist(Pprime, V, 'cosine'), axis=1)
    inValid = np.setdiff1d(np.arange(NV), associate)
    V[inValid] = np.random.rand(len(inValid), m)*np.max(Pprime, axis=0)
    return V

def truncation(P, N):
    """Truncates the population to a given population size"""
    Choose = np.ones(len(P.dec), dtype=bool)
    Cosine = 1-distance.cdist(P.obj, P.obj, 'cosine')
    np.fill_diagonal(Cosine, 0)
    while np.sum(Choose) > N:
        Remain = np.where(Choose)[0]
        Temp = np.sort(-Cosine[Remain][:,Remain], axis=1)
        Rank = np.lexsort(np.transpose(Temp)[::-1])
        Choose[Remain[Rank[0]]] = False
    S = population(P.dec[Choose], P.obj[Choose])
    return S
