"""
Reference set adaptation method based on niching and pair-potential functions.
"""

import numpy as np
from scipy.spatial import distance

from Public.Population import population
from Public.EfficientNonDominatedSort import efficientNonDominatedSort

def nichingPairPotentialAdaptationMethod(A, Q, P, Wadapt, W, N, zmin, fstep, flast, generations, max_generations, flag, znadir=None, scale=False, update=False, T=None, B=None, utility=None):
    """Main framework of the reference set adaptation method based on niching and pair-potential"""
    if (generations <= flast*max_generations):
        if znadir is None:
            Fronts = efficientNonDominatedSort(P.obj)
            zmax = np.max(P.obj[Fronts[0]], axis=0)
        else:
            zmax = np.copy(znadir)
        zmean = np.mean(A, axis=0)
        zstd = np.std(A, axis=0)
        zppf = zmean+6*zstd
        A, S, Wvalid = nichingPairPotentialArchive(A, Q, W, N, zmin, zmax, zppf, flag)
        if np.floor(generations % (fstep*max_generations)) == 0:
            Wadapt = adaptReferenceSet(A, S, Wvalid, W, N, zmin, zmax, scale)
            if update is True:
                P, B = updateBestAndNeighbors(P, Wadapt, N, T, zmin, zmax, scale, utility)
    return A, Wadapt, P, B

def nichingPairPotentialArchive(A, Q, W, N, zmin, zmax, zppf, flag):
    """Updates external archive based on niching and pair-potential selection"""
    A = np.vstack((A, Q.obj))
    unique = np.sort(np.unique(np.around(A, 6), return_index=True, axis=0)[1])
    A = A[unique]
    Fronts = efficientNonDominatedSort(A)
    A = A[Fronts[0]]
    S = []
    Wvalid = []
    if len(A) >= N:
        denom = zmax-zmin
        denom[denom == 0] = 1e-12
        Aprime = (A-zmin)/denom
        pi, d = associate(Aprime, W)
        rho = nicheCount(pi, W)
        A, S = nichingPairPotentialSelection(A, N, rho, pi, d, zmin, zppf, flag)
        Wvalid = W[rho > 0]
    return A, S, Wvalid

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

def nichingPairPotentialSelection(A, N, rho, pi, d, zmin, zppf, flag):
    """Selects N solutions from A using niching and pair-potential selection"""
    Choose = np.zeros(len(A), dtype=bool)
    J = np.where(rho > 0)[0]
    for j in J:
        I = np.where(pi == j)[0]
        np.random.shuffle(I)
        s = I[np.argmin(d[I])]
        Choose[s] = True
    S = A[Choose]
    selected = len(S)
    if (selected == N):
        A = np.copy(S)
    else:
        A = np.vstack((S, A[~Choose][np.all(A[~Choose] <= zppf, axis=1)]))
        zmax_arch = np.max(A, axis=0)
        denom = zmax_arch-zmin
        denom[denom == 0] = 1e-12
        Aprime = (A-zmin)/denom
        Diss = dissimilarityMatrix(Aprime, flag)
        Memo = np.sum(Diss, axis=1)
        while (len(A) > N):
            C = np.abs(Memo)
            I = np.arange(selected, len(C))
            np.random.shuffle(I)
            worst = I[np.argmax(C[I])]
            Diss = np.delete(Diss, worst, axis=0)
            Diss = np.delete(Diss, worst, axis=1)
            Memo = np.sum(Diss, axis=1)
            A = np.delete(A, worst, axis=0)
    return A, S

def dissimilarityMatrix(A, flag):  
    """Calculates dissimilarity matrix using a given pair-potential function"""
    if (flag == 1):
        m = np.shape(A)[1]
        Diss = dissimilarityMatrixRSE(A, m-1)
    elif (flag == 2):
        Diss = dissimilarityMatrixGAE(A, 512)
    elif (flag == 3):
        Diss = dissimilarityMatrixCOU(A)
    elif (flag == 4):
        Diss = dissimilarityMatrixPT(A, 5, 3, 0.02)
    elif (flag == 5):
        Diss = dissimilarityMatrixMPT(A, 1, 25)
    elif (flag == 6):
        Diss = dissimilarityMatrixKRA(A, 5, 3, 0.02)
    return Diss

def dissimilarityMatrixRSE(A, s):
    """Returns dissimilarity matrix using Riesz s-energy"""
    d = distance.pdist(A, 'euclidean')
    denom = d**s
    denom[denom == 0] = 1e-12
    d = 1/denom
    return distance.squareform(d)

def dissimilarityMatrixGAE(A, alpha):
    """Returns dissimilarity matrix using Gaussian alpha-energy"""
    d = distance.pdist(A, 'euclidean')
    d = np.e**(-alpha*(d**2))
    return distance.squareform(d)

def dissimilarityMatrixCOU(A):
    """Returns dissimilarity matrix using Coulomb's law"""
    k = 1/(4*np.pi*8.854e-12)
    norm = np.linalg.norm(A, axis=1)
    V = np.outer(norm, norm)
    np.fill_diagonal(V, 0)
    v = distance.squareform(V)
    d = distance.pdist(A, 'euclidean')
    denom = d**2
    denom[denom == 0] = 1e-12
    d = k*v/denom
    return distance.squareform(d)

def dissimilarityMatrixPT(A, V1, V2, alpha):
    """Returns dissimilarity matrix using Pösch-Teller Potential"""
    d = distance.pdist(A, 'euclidean')
    denom1 = np.sin(alpha*d)**2
    denom1[denom1 == 0] = 1e-12
    denom2 = np.cos(alpha*d)**2
    denom2[denom2 == 0] = 1e-12
    d = V1/denom1+V2/denom2
    return distance.squareform(d)

def dissimilarityMatrixMPT(A, D, alpha):
    """Returns dissimilarity matrix using Modified Pösch-Teller Potential"""
    d = distance.pdist(A, 'euclidean')
    d = -D/(np.cosh(alpha*d)**2)
    return distance.squareform(d)

def dissimilarityMatrixKRA(A, V1, V2, alpha):
    """Returns dissimilarity matrix using Kratzer Potential"""
    d = distance.pdist(A, 'euclidean')
    denom = np.copy(d)
    denom[denom == 0] = 1e-12
    d = V1*(((d-(1/alpha))/denom)**2)+V2
    return distance.squareform(d)

def adaptReferenceSet(A, S, Wvalid, W, N, zmin, zmax, scale):
    """Returns an adapted reference set of weight vectors"""
    zmax_arch = np.max(A, axis=0)
    r = zmax_arch-zmin
    epsilon = 1e-3
    if len(A) == N and np.all(r > epsilon):
        denom = zmax-zmin
        denom[denom == 0] = 1e-12
        V = (A[len(S):]-zmin)/denom
        denom = np.sum(V, axis=1)
        denom[denom == 0] = 1e-12
        Vweight = V/denom[:,np.newaxis]
        Wadapt = np.vstack((Wvalid, Vweight))
    else:
        Wadapt = np.copy(W)
    if scale is True:
        Wscale = Wadapt*(zmax-zmin)
        denom = np.sum(Wscale, axis=1)
        denom[denom == 0] = 1e-12
        Wadapt = Wscale/denom[:,np.newaxis]
    return Wadapt

def updateBestAndNeighbors(P, Wadapt, N, T, zmin, zmax, scale, utility):
    """Updates best solution and neighbors for each subproblem"""
    if scale is True:
        denom = zmax-zmin
        denom[denom == 0] = 1e-12
        V = Wadapt/denom
        denom = np.sum(V, axis=1)
        denom[denom == 0] = 1e-12
        Vweight = V/denom[:,np.newaxis]
        B = np.argsort(distance.cdist(Vweight, Vweight, 'euclidean'), axis=1)[:,:T]
    else:
        B = np.argsort(distance.cdist(Wadapt, Wadapt, 'euclidean'), axis=1)[:,:T]
    Pupdate = population([], [])
    for i in range(0, N):
        fitness = utility(P.obj, Wadapt[i], zmin)
        I = np.arange(N)
        np.random.shuffle(I)
        best = I[np.argmin(fitness[I])]
        Pupdate.dec = np.copy(P.dec[best]) if len(Pupdate.dec) == 0 else np.vstack((Pupdate.dec, P.dec[best]))
        Pupdate.obj = np.copy(P.obj[best]) if len(Pupdate.obj) == 0 else np.vstack((Pupdate.obj, P.obj[best]))
    return Pupdate, B
