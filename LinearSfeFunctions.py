import numpy as np
from numpy import array, diag, sqrt, transpose, ones_like, zeros_like, eye, abs
from numpy.linalg import inv, eig
from numpy.random import rand

def SimultaneouslyDiagonalize(X,Y):
    e,E = eig(Y)
    W = diag(sqrt(1/e))
    
    D_ = W @ transpose(E) @ X @ E @ W
    V  = eig(D_)[1]
    B  = transpose(E @ W @ V)
    return diag(diag(B @ X @ transpose(B))), diag(diag(B @ Y @ transpose(B))), B

def DiagonalizingBasis(X,Y):
    e,E = eig(Y)
    W = diag(sqrt(1/e))
    
    D_ = W @ transpose(E) @ X @ E @ W
    V  = eig(D_)[1]
    B  = transpose(E @ W @ V)
    return B


def SymmetricSfeSlope(gamma,delta,n):
    #print(gamma, delta, n)
    if n ==1:
        return delta/(1+delta*gamma)
    else:
        return (n-2-gamma*delta + sqrt((n-2-gamma*delta)**2 +4*(n-1)*gamma*delta))/(2*(n-1)*gamma)


def SymmetricLinearSFE(C,D,n):
    # C and D are marginal cost and demand slopes [floats], n is number of symmetric suppliers [int]
    X = DiagonalizingBasis(inv(C),D)
    C_tilde = diag(inv(X @ inv(C) @ transpose(X)))
    D_tilde = diag(X @ D @ transpose(X))
    #print(C_tilde)
    #print(D_tilde)
    #print(n*ones_like(C_tilde))
    Kappa = list(map(SymmetricSfeSlope,C_tilde,D_tilde,n*ones_like(C_tilde)))
    #print(diag(Kappa))
    #print("X:",X)
    return inv(X) @ diag(Kappa) @ inv(transpose(X))


def CournotSlope(gamma,delta,n):
    return(delta/(1+delta*gamma))

def CournotSchedule(C,D,n):
    # C and D are marginal cost and demand slopes [floats], n is number of symmetric suppliers [int]
    X = DiagonalizingBasis(inv(C),D)
    C_tilde = diag(inv(X @ inv(C) @ transpose(X)))
    D_tilde = diag(X @ D @ transpose(X))
    #print(C_tilde)
    #print(D_tilde)
    #print(n*ones_like(C_tilde))
    Kappa = list(map(CournotSlope,C_tilde,D_tilde,n*ones_like(C_tilde)))
    #print(diag(Kappa))
    #print("X:",X)
    return inv(X) @ diag(Kappa) @ inv(transpose(X))

# For testing purposes: random 2x2 positive-definite matrices
def RandomPosDef(n=2):
    A = rand(n,n)
    return A @ transpose(A)

from scipy.stats import wishart

def WishartRandomPosDef(n=2):
    return wishart.rvs(n,eye(n))

# Q: What is the distribution of these matrices?

#print('Oi!')

def RoundOut(x,precision = 2e-15):
    # Rounds all values in $x less than $precision to 0
    return (abs(x)>precision)*x