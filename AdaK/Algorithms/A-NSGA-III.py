"""
A-NSGA-III.

H. Jain and K. Deb, “An Evolutionary Many-Objective Optimization Algorithm 
Using Reference-Point Based Nondominated Sorting Approach, Part II: Handling 
Constraints and Extending to an Adaptive Approach,” in IEEE Transactions on 
Evolutionary Computation, vol. 18, no. 4, pp. 602–622, 2014.

J. Blank and K. Deb, "pymoo: Multi-Objective Optimization in Python," in IEEE 
Access, vol. 8, pp. 89497-89509, 2020.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np
import warnings
from numpy.linalg import LinAlgError

from Public.Population import population
from Public.UploadTestProblem import uploadTestProblem
from Public.UniformPoints import uniformPoints
from Public.RandomPopulation import randomPopulation
from Public.GenerateOffspring import generateOffspring
from Public.EfficientNonDominatedSort import efficientNonDominatedSort

def main(N, problem, m, max_generations):
    """Runs main framework of A-NSGA-III"""
    parameters, evaluate = uploadTestProblem(problem)
    n, lb, ub = parameters(m)
    W = uniformPoints(N, m)
    N = len(W)
    pc, nc = 1, 30
    pm, nm = 1/n, 20
    P = randomPopulation(N, n, m, lb, ub, evaluate)
    zmin = np.min(P.obj, axis=0)
    zworst = np.max(P.obj, axis=0)
    extreme = None
    generations = 0
    
    W = W[np.lexsort(np.transpose(W)[::-1])]
    interval = W[0,-1]-W[1,-1]
    
    Data_gen = []
    
    while generations < max_generations:
        M = matingSelection(P, N)
        Q = generateOffspring(M, N, m, lb, ub, pc, nc, pm, nm, evaluate)
        R = population(np.vstack((P.dec, Q.dec)), np.vstack((P.obj, Q.obj)))
        zmin = np.min(np.vstack((zmin, R.obj)), axis=0)
        zworst = np.max(np.vstack((zworst, R.obj)), axis=0)
        P, extreme = environmentalSelection(R, W, N, zmin, zworst, extreme)
        generations += 1
        W = adaptive(P, W, N, interval, zmin, zworst, extreme)
        
        Data_gen.append(P.obj)
        
    return P, Data_gen

def matingSelection(P, N):
    """Selects random parent population"""
    O = N+1 if N%2 == 1 else N
    indexes = np.array([np.random.choice(N, 2, replace=False) for i in range(0, O//2)])
    M = P.dec[np.hstack((indexes[:,0], indexes[:,1]))]
    return M

def environmentalSelection(R, W, N, zmin, zworst, extreme):
    """Returns population with best individuals"""
    unique = np.sort(np.unique(np.around(R.obj, 6), return_index=True, axis=0)[1])
    R.dec = R.dec[unique]
    R.obj = R.obj[unique]
    Fronts = efficientNonDominatedSort(R.obj)
    P, Fl = findCriticalFront(R, Fronts, N)
    NDS = R.obj[Fronts[0]]
    zmax_pop = np.max(R.obj, axis=0)
    zmax_front = np.max(NDS, axis=0)
    N1 = len(P.dec)
    if N1 < N:
        K = N-N1
        S = np.copy(Fl.obj) if N1 == 0 else np.vstack((P.obj, Fl.obj))
        Sn, extreme = normalize(S, NDS, zmin, zmax_pop, zmax_front, zworst, extreme)
        pi, d = associate(Sn, W)
        rho = nicheCount(pi[:N1], W)
        P = niching(K, rho, pi[N1:], d[N1:], Fl, P)
    return P, extreme

def findCriticalFront(R, Fronts, N):
    """Returns population in best fronts and population in critical front"""
    P = population([], [])
    for front in Fronts:
        if len(P.dec)+len(front) > N:
            Fl = population(R.dec[front], R.obj[front])
            break
        else:
            P.dec = R.dec[front] if len(P.dec) == 0 else np.vstack((P.dec, R.dec[front]))
            P.obj = R.obj[front] if len(P.obj) == 0 else np.vstack((P.obj, R.obj[front]))
    return P, Fl

def normalize(S, NDS, zmin, zmax_pop, zmax_front, zworst, extreme):
    """Normalization procedure"""
    m = np.shape(S)[1]
    weights = np.eye(m)
    weights[weights == 0] = 1e6
    Sadd = NDS
    if extreme is not None:
        Sadd = np.concatenate([extreme, Sadd], axis=0)
    Sprime = Sadd-zmin
    Sprime[Sprime < 1e-3] = 0
    Sasf = np.max(Sprime*weights[:,None,:], axis=2)
    I = np.argmin(Sasf, axis=1)
    extreme = Sadd[I, :]
    try:
        M = extreme-zmin
        b = np.ones(m)
        plane = np.linalg.solve(M, b)
        warnings.simplefilter("ignore")
        intercepts = 1/plane
        znadir = zmin+intercepts
        if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6):
            raise LinAlgError()
        b = znadir > zworst
        znadir[b] = zworst[b]
    except LinAlgError:
        znadir = zmax_front
    b = znadir-zmin <= 1e-6
    znadir[b] = zmax_pop[b]
    denom = znadir-zmin
    denom[denom == 0] = 1e-12
    Sn = (S-zmin)/denom
    return Sn, extreme

def associate(A, W):
    """Returns closest reference point and its distance for each solution"""
    D = perpendicularDistance(A, W)
    pi = np.argmin(D, axis=1)
    d = D[np.arange(len(A)),pi]
    return pi, d

def perpendicularDistance(A, W):
    """Returns fitness matrix using perpendicular distance"""
    U = np.tile(W, (len(A), 1))
    V = np.repeat(A, len(W), axis=0)
    Unorm = np.linalg.norm(U, axis=1)
    Proj_scalar = np.sum(V*U, axis=1)/Unorm
    Proj = Proj_scalar[:,None]*U/Unorm[:,None]
    val = np.linalg.norm(Proj-V, axis=1)
    D = np.reshape(val, (len(A), len(W)))
    return D

def nicheCount(pi, W):
    """Return niche count of each reference point"""
    rho = np.zeros(len(W), dtype=int)
    index, count = np.unique(pi, return_counts=True)
    rho[index] = count
    return rho

def niching(K, rho, pi, d, Fl, P):
    """Selects K solutions from Fl to complete P using the niching procedure"""
    survivors = []
    mask = np.full(len(Fl.dec), True)
    while len(survivors) < K:
        K1 = K-len(survivors)
        pi_next = np.unique(pi[mask])
        rho_next = rho[pi_next]
        Jmin = pi_next[np.where(rho_next == np.min(rho_next))[0]]
        Jmin = Jmin[np.random.permutation(len(Jmin))[:K1]]
        for j in Jmin:
            I = np.where(np.logical_and(pi == j, mask))[0]
            np.random.shuffle(I)
            s = I[np.argmin(d[I])] if rho[j] == 0 else I[0]
            mask[s] = False
            survivors.append(int(s))
            rho[j] += 1
    P.dec = Fl.dec[survivors] if len(P.dec) == 0 else np.vstack((P.dec, Fl.dec[survivors]))
    P.obj = Fl.obj[survivors] if len(P.obj) == 0 else np.vstack((P.obj, Fl.obj[survivors]))
    return P

def adaptive(P, W, N, interval, zmin, zworst, extreme):
    """Performs the reference set adaptation method from A-NSGA-III"""
    m = np.shape(P.obj)[1]
    Fronts = efficientNonDominatedSort(P.obj)
    NDS = P.obj[Fronts[0]]
    zmax_pop = np.max(P.obj, axis=0)
    zmax_front = np.max(NDS, axis=0)
    An, extreme = normalize(P.obj, NDS, zmin, zmax_pop, zmax_front, zworst, extreme)
    pi, d = associate(An, W)
    rho = nicheCount(pi, W)
    Wold = np.array([])
    while (any(rho >= 2) and not(np.array_equal(Wold, W))):
        Wold = np.copy(W)
        for i in np.flatnonzero(rho >= 2):
            p = np.matlib.repmat(W[i,:], m, 1)-interval/m
            p[np.eye(m, dtype=bool)] += interval
            W = np.vstack((W, p))
        W = W[np.any(W < 0, axis=1) == False]
        u, index = np.unique(np.around(W, 4), return_index=True, axis=0)
        W = W[np.sort(index)]
        pi, d = associate(An, W)
        rho = nicheCount(pi, W)
    index = np.intersect1d(np.arange(N, len(W)), np.where(rho == 0)[0])
    if (len(index) > 0):
        W = np.delete(W, index, axis=0)
    return W
