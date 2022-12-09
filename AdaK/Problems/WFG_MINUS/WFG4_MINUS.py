"""
WFG4_MINUS test problem.

S. Huband, L. Barone, L. While, and P. Hingston, "A Scalable Multi-objective 
Test Problem Toolkit," in Evolutionary Multi-Criterion Optimization, 
pp. 280-295, 2005.

H. Ishibuchi, Y. Setoguchi, H. Masuda, and Y. Nojima, "Performance of 
Decomposition-Based Many-Objective Algorithms Strongly Depends on Pareto Front 
Shapes," in IEEE Transactions on Evolutionary Computation, vol. 21, no. 2, 
pp. 169-190, 2017.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np

def parameters(m):
    """Returns number of decision variables, lower bounds, and upper bounds"""
    k = m-1
    l = 24-(m-1)
    n = k+l
    lb = np.zeros(n)
    ub = np.arange(2.0, 2*n+1, 2)
    return n, lb, ub

def evaluate(P, m):
    """Evaluates a population for the WFG4_MINUS test problem"""
    N, n = np.shape(P)
    k = m-1
    D = 1
    A = np.ones(m-1)
    S = np.arange(2.0, 2*m+1, 2)
    
    z01 = P/np.tile(np.arange(2.0, 2*n+1, 2), (N, 1))
    
    t1 = np.zeros((N, n))
    t1 = s_multi(z01, 30, 10, 0.35)
    
    t2 = np.zeros((N, m))
    for i in range(0, m-1):
        t2[:,i] = r_sum(t1[:,i*k//(m-1):(i+1)*k//(m-1)], np.ones(k//(m-1)))
    t2[:,m-1] = r_sum(t1[:,k:], np.ones(n-k))
    
    x = np.zeros((N, m))
    for i in range(0, m-1):
        x[:,i] = np.maximum(t2[:,m-1], A[i])*(t2[:,i]-0.5)+0.5
    x[:,m-1] = t2[:,m-1]
    
    h = concave(x)
    
    return -(np.tile(D*x[:,m-1][:,np.newaxis], (1, m))+np.tile(S, (N, 1))*h)

def s_multi(y, A, B, C):
    """Transformation function. Shift: Multi-modal"""
    out = (1+np.cos((4*A+2)*np.pi*(0.5-np.abs(y-C)/(2*(np.floor(C-y)+C))))+4*B*(np.abs(y-C)/(2*(np.floor(C-y)+C)))**2)/(B+2)
    return correct_to_01(out)

def r_sum(y, w):
    """Transformation function. Reduction: Weighted Sum"""
    out = np.sum(y*np.tile(w, (len(y), 1)), axis=1)/np.sum(w)
    return correct_to_01(out)

def concave(x):
    """Shape function. Concave"""
    N = len(x)
    out = np.fliplr(np.cumprod(np.hstack((np.ones((N, 1)), np.sin(x[:,:-1]*np.pi/2))), axis=1))*np.hstack((np.ones((N, 1)), np.cos(x[:,-2::-1]*np.pi/2)))
    return correct_to_01(out)

def correct_to_01(out):
    """Maintains output onto range using a given epsilon"""
    epsilon = 1.0e-10
    min_value = 0.0
    max_value = 1.0
    min_epsilon = min_value-epsilon
    max_epsilon = max_value+epsilon
    out[np.logical_and(out < min_value, out >= min_epsilon)] = min_value
    out[np.logical_and(out > max_value, out <= max_epsilon)] = max_value
    return out
