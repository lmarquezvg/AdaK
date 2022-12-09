"""
Generate offspring using simulated binary crossover and polynomial mutation.

K. Deb and R. Agrawal, "Simulated binary crossover for continuous search 
space," in Complex Systems, vol. 9, no. 2, pp. 115–148, 1995.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np

from Public.Population import population

def generateOffspring(M, N, m, lb, ub, pc, nc, pm, nm, evaluate):
    """Generates offspring population from mating pool"""
    Q = population([], [])
    MA, MB = M[:len(M)//2], M[len(M)//2:]
    Q.dec = simulatedBinaryCrossover(MA, MB, N, lb, ub, pc, nc)
    Q.dec = polynomialMutation(Q.dec, lb, ub, pm, nm)
    Q.obj = evaluate(Q.dec, m)
    if (N == 1):
        Q.dec = Q.dec[0]
        Q.obj = Q.obj[0]
    return Q

def simulatedBinaryCrossover(MA, MB, N, lb, ub, pc, nc):
    """Generates an offspring population using simulated binary crossover"""
    O, n = np.shape(MA)
    beta = np.zeros((O, n))
    mu = np.random.rand(O, n)
    beta[mu<=0.5] = (2*mu[mu<=0.5])**(1/(nc+1))
    beta[mu>0.5] = (2-2*mu[mu>0.5])**(-1/(nc+1))
    beta *= (-1)**np.random.randint(0, 2, (O, n))
    beta[np.random.rand(O, n) <= 0.5] = 1
    beta[np.tile(np.random.rand(O, 1) > pc, (1, n))] = 1
    Q = np.vstack(((MA+MB)/2+beta*(MA-MB)/2, (MA+MB)/2-beta*(MA-MB)/2))
    if N%2 == 1:
        Q = np.delete(Q, len(Q)-1, axis=0) if np.random.rand() <= 0.5 else np.delete(Q, len(Q)//2-1, axis=0)
    lbext = np.tile(lb, (N, 1))
    ubext = np.tile(ub, (N, 1))
    Q = np.minimum(np.maximum(Q, lbext), ubext)
    return Q

def polynomialMutation(Q, lb, ub, pm, nm):
    """Mutates an offspring population using polynomial mutation"""
    N, n = np.shape(Q)
    mutate = np.random.rand(N, n) <= pm
    mu = np.random.rand(N, n)
    Qmut = np.copy(Q)
    lbext = np.tile(lb, (N, 1))
    ubext = np.tile(ub, (N, 1))
    temp = mutate & (mu <= 0.5)
    Qmut[temp] += (ubext[temp]-lbext[temp])*((2*mu[temp]+(1-2*mu[temp])*(1-(Qmut[temp]-lbext[temp])/(ubext[temp]-lbext[temp]))**(nm+1))**(1/(nm+1))-1)
    temp = mutate & (mu > 0.5)
    Qmut[temp] += (ubext[temp]-lbext[temp])*(1-(2*(1-mu[temp])+2*(mu[temp]-0.5)*(1-(ubext[temp]-Qmut[temp])/(ubext[temp]-lbext[temp]))**(nm+1))**(1/(nm+1)))
    Qmut = np.minimum(np.maximum(Qmut, lbext), ubext)
    return Qmut
