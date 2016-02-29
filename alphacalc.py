# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:31:20 2016

@author: bradc
"""
"""
Numerically solve the ode for the complex coefficients of the scalar solution in the 
quasistable regime using the recursion coefficients. Test up to L=10.
"""

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import gamma
from scipy.special import eval_jacobi
import math
from sympy import lambdify, jacobi
from sympy.functions.elementary.trigonometric import cos as symcos
from sympy.core import diff
from sympy.abc import t
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import *

###################################################################
###################################################################

"""
Read in values for X,Y,S from recursion relations.
"""

S = np.genfromtxt("d3S_L10.dat")
X = np.genfromtxt("d3X_L10.dat")
Y = np.genfromtxt("d3Y_L10.dat")

"""
Read in T,R from Mathematica output
"""

R = np.genfromtxt("Mathematica_R.dat")
T = np.genfromtxt("Mathematica_T.dat")


###################################################################
###################################################################
  
"""
Basis functions to be used in computing other tensors
"""
def w(n,d):
    if n<0:
        return float(d)
    else:
        return d+2.*n
     
def ks(i,d):
    return 2.*math.sqrt(math.factorial(i)*math.factorial(i+d-1))/gamma(i + 0.5*d)
    
def ep(i,d):
    f = diff(ks(i,d)*((symcos(t))**d)*jacobi(i,0.5*d -1,0.5*d,symcos(2*t)),t)
    return lambdify(t,f)
    
"""
Functions to extract the desired tensor elements from the recursion outputs.
"""
def Xval(i,j,k,l):
    temp = sorted([j,k,l],reverse=True)
    for row in range(X.shape[0]):
        if X[row][0]==i and X[row][1]==temp[0] and X[row][2]==temp[1] and X[row][3]==temp[2]:
            return X[row][4] 
    print("X[%d][%d][%d][%d] not found, returning 0" % (i,j,k,l))
    return 0
    

def Yval(i,j,k,l):
    temp = sorted([k,l],reverse=True)
    for row in range(Y.shape[0]):
        if Y[row][0]==i and Y[row][1]==j and Y[row][2]==temp[0] and Y[row][3]==temp[1]:
            return Y[row][4]
    print("Y[%d][%d][%d][%d] not found, returning 0" % (i,j,temp[0],temp[1]))
    return 0
    

def Sval(i,j,k,l):
    for row in range(S.shape[0]):
        if S[row][0]==i and S[row][1]==j and S[row][2]==k and S[row][3]==l:
            if str(S[row][4]) == "nan":
                return 0
            else:
                return S[row][4]
    print("S[%d][%d][%d][%d] not found" % (i,j,k,l))
    return 0

      
def Rval(i,j):
    for row in range(R.shape[0]):
        if R[row][0]==i and R[row][1]==j:
            if str(R[row][2]) == "nan":
                return 0
            else:
                return R[row][2]

        
def Tval(i):
    for row in range(T.shape[0]):
        if T[row][0]==i:
            return T[row][1]


"""
These tensors will be calculated by integrals elsewhere before being used; for the test case,
generating the values as needed is sufficient.
""" 
    
def W00(i,j,k,l,d):
    return dblquad(lambda x,y: (ks(i,d)*((math.cos(x))**d)*eval_jacobi(i,0.5*d -1,0.5*d,math.cos(2*x))) \
    * (ks(j,d)*((math.cos(x))**d)*eval_jacobi(j,0.5*d -1,0.5*d,math.cos(2*x))) * math.sin(x)*math.cos(x) \
    * (ks(l,d)*((math.cos(y))**d)*eval_jacobi(l,0.5*d -1,0.5*d,math.cos(2*y))) \
    * (ks(k,d)*((math.cos(y))**d)*eval_jacobi(k,0.5*d -1,0.5*d,math.cos(2*y))) * (math.tan(y))**(d-1.), \
    0,math.pi/2, lambda x:0, lambda x:x)[0]
    
def W10(i,j,k,l,d):
    return dblquad(lambda x,y: (ep(i,d)(x))*(ep(j,d)(x)) * math.sin(x)*math.cos(x) \
    * (ks(l,d)*((math.cos(y))**d)*eval_jacobi(l,0.5*d -1,0.5*d,math.cos(2*y))) \
    * (ks(k,d)*((math.cos(y))**d)*eval_jacobi(k,0.5*d -1,0.5*d,math.cos(2*y))) * (math.tan(y))**(d-1.), \
    0,math.pi/2, lambda x:0, lambda x:x)[0]
    
def A(i,j,d):
    return quad(lambda x: (ks(i,d)*((math.cos(x))**d)*eval_jacobi(i,0.5*d -1,0.5*d,math.cos(2*x))) \
    * (ks(j,d)*((math.cos(x))**d)*eval_jacobi(j,0.5*d -1,0.5*d,math.cos(2*x))) * math.sin(x)*math.cos(x), \
    0, math.pi/2.)[0]
    
def V(i,j,d):
    return quad(lambda x: (ep(i,d)(x))*(ep(j,d)(x))*math.sin(x)*math.cos(x),0,math.pi/2.)[0]
    
#def T(i,d):
#    return ((w(i,d)**2)*(Xval(i,i,i,i))/2. + 3*(Yval(i,i,i,i))/2. + 2*(w(i,d)**4)*W00(i,i,i,i,d) + 2*(w(i,d)**2)*W10(i,i,i,i,d) \
#    - (w(i,d)**2)*(A(i,i,d) + (w(i,d)**2)*V(i,i,d)))
    
#def R(i,j,d):
#    return (((w(i,d)**2 + w(j,d)**2)/(w(j,d)**2 - w(i,d)**2))*((w(j,d)**2)*Xval(i,j,j,i) - (w(i,d)**2)*Xval(j,i,i,j))/2. \
#                + 2*((w(j,d)**2)*Yval(i,j,i,j) - (w(i,d)**2)*Yval(j,i,j,i))/(w(j,d)**2 - w(i,d)**2) \
#                + (Yval(i,i,j,j) + Yval(j,j,i,i))/2. \
#                +(w(i,d)**2)*(w(j,d)**2)*(Xval(i,j,j,i) - Xval(j,i,j,i))/(w(j,d)**2 - w(i,d)**2) \
#                + (w(i,d)**2)*(w(j,d)**2)*(W00(j,j,i,i,d) + W00(i,i,j,j,d)) \
#                + (w(i,d)**2)*(W10(j,j,i,i,d)) + (w(j,d)**2)*(W10(i,i,j,j,d)) \
#                - (w(j,d)**2)*(A(i,i,d) + (w(i,d)**2)*V(i,i,d)))
   
"""
Define initial values of a and b based on a0 = 1.0, a1 = 0.1 and the first two terms
of the QP mode equation
"""

# Initial values for alpha_0 and alpha_1; maximum number N = j_max; dimension d
a0 = 1.0
a1 = 0.1
N = 3
d=3

# Initialize alpha
alpha = np.ones(N)        
alpha[0] = a0
alpha[1] = a1        
    

 
"""    
Functions for solving TTF coefficients
"""

def b(i,d):
    if i == 0:
        return -2.*Tval(0)/w(0,d)
    if i == 1:
        return -1.0*(4.0*Tval(1)*alpha[1]**3 + 4*Rval(1,0)*alpha[1] + 4.0*Sval(1,0,0,0)*alpha[0]**3)/(2.0*w(1,d)*alpha[1])
    else:
        return b(0,d)+(b(1,d)-b(0,d))*i*1.0

def f(x):
    return 4.0*Tval(2)*x**3 + 4.0*(Rval(2,0)*x + Rval(2,1)*x*(0.1)**2) + 4.*(Sval(2,0,0,0) + 0.1*(Sval(2,0,0,1) \
    + Sval(2,0,1,0) + Sval(2,1,0,0)) + (0.1**2)*(Sval(2,0,1,1) + Sval(2,1,0,1) + Sval(2,1,1,0)) \
    + x*(Sval(2,0,0,2)+Sval(2,0,2,0) + Sval(2,2,0,0))) + 2.*w(2,3)*b(2,3)*x


# Use the QP equation (14) from arXiv:1507.08261
def F(alpha):
    F = np.zeros(N)
    for i in range(N):
        for j in range(i+1):
            for k in range(j+1):
                for l in range(k+1):
                    if j+k+l <= i:
                        F[i]= 4.*Tval(i)*alpha[i]**3 + 4.*Rval(i,j)*alpha[j]*alpha[i]**2 + 4.*Sval(i,j,k,l)*alpha[j]*alpha[k]*alpha[l] \
                        + 2.*w(i,d)*b(i,d)*alpha[i]
                    else:
                        F[i]= 4.*Tval(i)*alpha[i]**3 + 4.*Rval(i,j)*alpha[j]*alpha[i]**2 + 2.*w(i,d)*b(i,d)*alpha[i]
    return F


"""
Solve for TTF coefficients
"""

print(newton_krylov(F,alpha,method='minres',verbose="True"))









###################################################################
###################################################################

print("X[1][0][1][0] =", Xval(1,0,1,0))
print("Y[1][0][0][1] =", Yval(1,0,0,1))
print("S[1][0][0][0] =", Sval(1,0,0,0))
print("T[1] =", Tval(2))
print("R[1][0] =", Rval(1,1))
print("b1 =", b(1,3))
print("b0 =", b(0,3))

###################################################################
###################################################################
