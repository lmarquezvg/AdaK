"""
WFG3 test problem.

S. Huband, L. Barone, L. While, and P. Hingston, "A Scalable Multi-objective 
Test Problem Toolkit," in Evolutionary Multi-Criterion Optimization, 
pp. 280-295, 2005.

Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for 
evolutionary multi-objective optimization," in IEEE Computational Intelligence 
Magazine, vol. 12, no. 4, pp. 73-87, 2017.
"""

import numpy as np

def parameters(m):
    """Returns number of decision variables, lower bounds, and upper bounds"""
    k = m-1
    l = 24-(m-1)
    if not l % 2 == 0:
        l -= 1
    n = k+l
    lb = np.zeros(n)
    ub = np.arange(2.0, 2*n+1, 2)
    return n, lb, ub

def evaluate(P, m):
    """Evaluates a population for the WFG3 test problem"""
    N, n = np.shape(P)
    k = m-1
    D = 1
    A = np.hstack((np.ones(1), np.zeros(m-2)))
    S = np.arange(2.0, 2*m+1, 2)
    
    z01 = P/np.tile(np.arange(2.0, 2*n+1, 2), (N, 1))
    
    t1 = np.zeros((N, n))
    t1[:,:k] = z01[:,:k]
    t1[:,k:] = s_linear(z01[:,k:], 0.35)
    
    t2 = np.zeros((N, k+(n-k)//2))
    t2[:,:k] = t1[:,:k]
    for i in range(k, k+(n-k)//2):
        t2[:,i] = r_nonsep(t1[:,k+2*(i+1-k)-2:k+2*(i+1-k)], 2)
    
    t3 = np.zeros((N, m))
    for i in range(0, m-1):
        t3[:,i] = r_sum(t2[:,i*k//(m-1):(i+1)*k//(m-1)], np.ones(k//(m-1)))
    t3[:,m-1] = r_sum(t2[:,k:], np.ones((n-k)//2))
    
    x = np.zeros((N, m))
    for i in range(0, m-1):
        x[:,i] = np.maximum(t3[:,m-1], A[i])*(t3[:,i]-0.5)+0.5
    x[:,m-1] = t3[:,m-1]
    
    h = linear(x)
    
    return np.tile(D*x[:,m-1][:,np.newaxis], (1, m))+np.tile(S, (N, 1))*h

def s_linear(y, A):
    """Transformation function. Shift: Linear"""
    out = np.abs(y-A)/np.abs(np.floor(A-y)+A)
    return correct_to_01(out)

def r_nonsep(y, A):
    """Transformation function. Reduction: Non-separable"""
    N, m = np.shape(y)
    sum1 = np.zeros(N)
    for j in range(0, m):
        sum2 = np.zeros(N)
        for k in range(0, A-1):
            sum2 += np.abs(y[:,j]-y[:,(1+j+k)%m])
        sum1 += y[:,j]+sum2
    out = sum1/(m*np.ceil(A/2)*(1+2*A-2*np.ceil(A/2))/A)
    return correct_to_01(out)

def r_sum(y, w):
    """Transformation function. Reduction: Weighted Sum"""
    out = np.sum(y*np.tile(w, (len(y), 1)), axis=1)/np.sum(w)
    return correct_to_01(out)

def linear(x):
    """Shape function. Linear"""
    N = len(x)
    out = np.fliplr(np.cumprod(np.hstack((np.ones((N, 1)), x[:,:-1])), axis=1))*np.hstack((np.ones((N, 1)), 1-x[:,-2::-1]))
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
