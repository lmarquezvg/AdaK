"""
Efficient Nondominated Sort.

X. Zhang, Y. Tian, R. Cheng, and Y. Jin, "An Efficient Approach to Nondominated 
Sorting for Evolutionary Multiobjective Optimization," in IEEE Transactions on 
Evolutionary Computation, vol. 19, no. 2, pp. 201-213, 2015.
"""

import numpy as np

def efficientNonDominatedSort(A):
    """Returns indexes of individuals divided by fronts"""
    F = [[]]
    Isorted = np.lexsort(np.transpose(A)[::-1])
    for i in Isorted:
        k = sequentialSearch(i, A, F)
        if k <= len(F):
            F[k-1].append(i)
        else:
            F.append([i])
    for front in F:
        front.sort()
    return F

def sequentialSearch(i, A, F):
    """Performs the sequential search strategy"""
    x = len(F)
    k = 1
    while True:
        Dominated = False
        for j in F[k-1][::-1]:
            if all(A[j] <= A[i]) and any(A[j] < A[i]):
                Dominated = True
                break
        if Dominated == False:
            return k
        else:
            k += 1
            if k > x:
                return x+1
