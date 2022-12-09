"""
WFG9_MINUS test problem.

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
    """Evaluates a population for the WFG9_MINUS test problem"""
    N, n = np.shape(P)
    k = m-1
    D = 1
    A = np.ones(m-1)
    S = np.arange(2.0, 2*m+1, 2)
    
    z01 = P/np.tile(np.arange(2.0, 2*n+1, 2), (N, 1))
    
    t1 = np.zeros((N, n))
    for i in range(0, n-1):
        t1[:,i] = b_param(z01[:,i], r_sum(z01[:,i+1:], np.ones(n-i-1)), 0.98/49.98, 0.02, 50)
    t1[:,n-1] = z01[:,n-1]
    
    t2 = np.zeros((N, n))
    t2[:,:k] = s_decept(t1[:,:k], 0.35, 0.001, 0.05)
    t2[:,k:] = s_multi(t1[:,k:], 30, 95, 0.35)
    
    t3 = np.zeros((N, m))
    for i in range(0, m-1):
        t3[:,i] = r_nonsep(t2[:,i*k//(m-1):(i+1)*k//(m-1)], k//(m-1))
    t3[:,m-1] = r_nonsep(t2[:,k:], n-k)
    
    x = np.zeros((N, m))
    for i in range(0, m-1):
        x[:,i] = np.maximum(t3[:,m-1], A[i])*(t3[:,i]-0.5)+0.5
    x[:,m-1] = t3[:,m-1]
    
    h = concave(x)
    
    return -(np.tile(D*x[:,m-1][:,np.newaxis], (1, m))+np.tile(S, (N, 1))*h)

def b_param(y, y_prime, A, B, C):
    """Transformation function. Bias: Parameter Dependent"""
    out = y**(B+(C-B)*(A-(1-2*y_prime)*np.abs(np.floor(0.5-y_prime)+A)))
    return correct_to_01(out)

def s_decept(y, A, B, C):
    """Transformation function. Shift: Deceptive"""
    out = 1+(np.abs(y-A)-B)*(np.floor(y-A+B)*(1-C+(A-B)/B)/(A-B)+np.floor(A+B-y)*(1-C+(1-A-B)/B)/(1-A-B)+1/B)
    return correct_to_01(out)

def s_multi(y, A, B, C):
    """Transformation function. Shift: Multi-modal"""
    out = (1+np.cos((4*A+2)*np.pi*(0.5-np.abs(y-C)/(2*(np.floor(C-y)+C))))+4*B*(np.abs(y-C)/(2*(np.floor(C-y)+C)))**2)/(B+2)
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
