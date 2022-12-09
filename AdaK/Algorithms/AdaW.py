"""
AdaW.

M. Li and X. Yao, "What Weights Work for You? Adapting Weights for Any Pareto 
Front Shape in Decomposition-Based Evolutionary Multiobjective Optimisation," 
in Evolutionary Computation, vol. 28, no. 2, pp. 227-253, 2020.
"""

import numpy as np
from scipy.spatial import distance
from scipy.stats import mode

from Public.Population import population
from Public.UploadTestProblem import uploadTestProblem
from Public.UniformPoints import uniformPoints
from Public.RandomPopulation import randomPopulation
from Public.GenerateOffspring import generateOffspring
from Public.EfficientNonDominatedSort import efficientNonDominatedSort

def main(N, problem, m, max_generations):
    """Runs main framework of AdaW"""
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
    NA = 2*N
    Fronts = efficientNonDominatedSort(P.obj)
    A = population(P.dec[Fronts[0]], P.obj[Fronts[0]])
    
    Data_gen = []
    
    while generations < max_generations:
        for i in range(0, N):
            if np.random.rand() < delta:
                I = np.random.permutation(B[i])
            else:
                I = np.random.permutation(N)
            M = P.dec[I[:2]]
            y = generateOffspring(M, 1, m, lb, ub, pc, nc, pm, nm, evaluate)
            A.dec = np.vstack((A.dec, y.dec))
            A.obj = np.vstack((A.obj, y.obj))
            zmin = np.minimum(zmin, y.obj)
            offspring = ASF(y.obj, W[I], zmin)
            neighbors = ASF(P.obj[I], W[I], zmin)
            replace = I[np.where(offspring <= neighbors)[0][:nr]]
            P.dec[replace] = y.dec
            P.obj[replace] = y.obj
        generations += 1
        
        Fronts = efficientNonDominatedSort(A.obj)
        A.dec = A.dec[Fronts[0]]
        A.obj = A.obj[Fronts[0]]
        if len(A.dec) > NA:
            A = archiveMaintenance(A, NA)
        
        if np.floor(generations % (fstep*max_generations)) == 0 and generations <= flast*max_generations:
            P, W = weightAddition(P, A, W, zmin, T)
            P, W = weightDeletion(P, W, N, zmin)
            B = np.argsort(distance.cdist(W, W, 'euclidean'), axis=1)[:,:T]
        
        Data_gen.append(np.copy(P.obj))
        
    return P, Data_gen

def ASF(P, W, zmin):
    """Evaluates a population using the achievement scalarizing function"""
    Weights = np.copy(W)
    Weights[Weights < 1e-3] = 1e-3
    return np.max(np.abs(P-zmin)/Weights, axis=1)

def archiveMaintenance(A, NA):
    """Archive maintenance mechanism based on crowding degree"""
    m = np.shape(A.obj)[1]
    zmin_arch = np.min(A.obj, axis=0)
    zmax_arch = np.max(A.obj, axis=0)
    denom = zmax_arch-zmin_arch
    denom[denom == 0] = 1e-12
    Aprime = (A.obj-zmin_arch)/denom
    Diss = distance.cdist(Aprime, Aprime, 'euclidean')
    np.fill_diagonal(Diss, float('inf'))
    r = np.median(np.sort(Diss, axis=1)[:,np.minimum(m-1, np.shape(Diss)[1]-1)])
    R = np.minimum(Diss/r, 1)
    while len(A.dec) > NA:
        D = 1-np.prod(R, axis=1)
        worst = np.argmax(D)
        R = np.delete(R, worst, axis=0)
        R = np.delete(R, worst, axis=1)
        A.dec = np.delete(A.dec, worst, axis=0)
        A.obj = np.delete(A.obj, worst, axis=0)
    return A

def weightAddition(P, A, W, zmin, T):
    """Generates and finds the undeveloped and promising weights"""
    zmin_arch = np.min(A.obj, axis=0)
    zmax_arch = np.max(A.obj, axis=0)
    denom = zmax_arch-zmin_arch
    denom[denom == 0] = 1e-12
    Aprime = (A.obj-zmin_arch)/denom
    Pprime = (P.obj-zmin_arch)/denom
    Diss1 = distance.cdist(Aprime, Pprime, 'euclidean')
    Diss2 = distance.cdist(Aprime, Aprime, 'euclidean')
    r = np.median(np.sort(Diss2, axis=1)[:,1])
    und = np.sort(Diss1, axis=1)[:,0] >= r
    Aund = population(A.dec[und], A.obj[und])
    if len(Aund.dec) > 0:
        V = Aund.obj-zmin
        denom = np.sum(V, axis=1)
        denom[denom == 0] = 1e-12
        W1 = V/denom[:,np.newaxis]
        for i in range(0, len(W1)):
            Wcomb = np.vstack((W, W1[i]))
            Diss = distance.cdist(Wcomb, Wcomb, 'euclidean')
            np.fill_diagonal(Diss, float('inf'))
            Bcomb = np.argsort(Diss, axis=1)[:,:T]
            Pcomb = np.vstack((P.obj, Aund.obj[i]))
            I = Bcomb[-1]
            undeveloped = ASF(np.tile(Aund.obj[i], (len(I), 1)), W1[i], zmin)
            neighbors = ASF(Pcomb[I], W1[i], zmin)
            index = np.where(neighbors < undeveloped)[0]
            if len(index) == 0:
                W = np.vstack((W, W1[i]))
                P.dec = np.vstack((P.dec, Aund.dec[i]))
                P.obj = np.vstack((P.obj, Aund.obj[i]))
                I = Bcomb[-1]
                promising = ASF(Aund.obj[i], W[I], zmin)
                neighbors = ASF(P.obj[I], W[I], zmin)
                replace = I[np.where(promising < neighbors)[0]]
                P.dec[replace] = Aund.dec[i]
                P.obj[replace] = Aund.obj[i]
    return P, W

def weightDeletion(P, W, N, zmin):
    """Deletes the poorly-performed weights"""
    while len(P.dec) > N:
        ai, bi = np.unique(P.obj, return_index=True, return_inverse=True, axis=0)[1:]
        if len(ai) == len(bi):
            m = np.shape(P.obj)[1]
            zmin_pop = np.min(P.obj, axis=0)
            zmax_pop = np.max(P.obj, axis=0)
            denom = zmax_pop-zmin_pop
            denom[denom == 0] = 1e-12
            Pprime = (P.obj-zmin_pop)/denom
            Diss = distance.cdist(Pprime, Pprime, 'euclidean')
            np.fill_diagonal(Diss, float('inf'))
            r = np.median(np.sort(Diss, axis=1)[:,np.minimum(m-1, np.shape(Diss)[1]-1)])
            R = np.minimum(Diss/r, 1)
            while len(P.dec) > N:
                D = 1-np.prod(R, axis=1)
                worst = np.argmax(D)
                R = np.delete(R, worst, axis=0)
                R = np.delete(R, worst, axis=1)
                P.dec = np.delete(P.dec, worst, axis=0)
                P.obj = np.delete(P.obj, worst, axis=0)
                W = np.delete(W, worst, axis=0)
        else:
            I = np.where(bi == mode(bi)[0])[0]
            values = ASF(P.obj[I], W[I], zmin)
            worst = I[np.where(values == np.max(values))[0][0]]
            P.dec = np.delete(P.dec, worst, axis=0)
            P.obj = np.delete(P.obj, worst, axis=0)
            W = np.delete(W, worst, axis=0)
    return P, W
