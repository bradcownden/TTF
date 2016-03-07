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
from scipy.optimize import newton

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
    #print("S[%d][%d][%d][%d] not found" % (i,j,k,l))
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
a1 = 0.2
N = 3
d = 3


 
"""    
Functions for solving TTF coefficients
"""

# Function that returns the input values a0, a1 when i=0,1 and returns the variable
# being optimized when i>1
def alpha(i,x):
    if i==0:
        return a0
    if i==1:
        return a1
    if i>1:
        return x[i]
    if i<0:
        #print("Index error in alpha, called for i=%d" % i)
        return 0

# Use the QP equation (14) from arXiv:1507.08261
def system(x):
    F = np.zeros_like(x)
    for i in range(N):
        s = 0; r = 0
        for j in range(N):
            for k in range(N):
                if j+k-i<N:
                    s = s +2.*Sval(j,k,j+k-i,i)*alpha(j,x)*alpha(k,x)*alpha(j+k-i,x) 
            r = r + 2.*Rval(i,j)*alpha(i,x)*alpha(j,x)**2
        F[i] = 2.*Tval(i)*alpha(i,x)**3 + w(i,d)*(x[0]+i*(x[1]-x[0]))*alpha(i,x) + r + s
    return F
    


# Compute the energy per mode using (5) from arXiv:1507.08261
def energy(x):
    E = np.zeros_like(x)
    E[0] = 4*w(0,d)**2*a0**2
    E[1] = 4*w(1,d)**2*a1**2
    for i in range(2,len(x)):
        E[i] = 4.*w(i,d)**2*x[i]**2
    return E


###################################################################
###################################################################


"""
Solve for TTF coefficients
"""
ainits = np.ones(N)
sol = newton_krylov(system,ainits)
print("Loop of QP equation [beta_0, beta_1, alpha_2] = [%.10e, %.10e, %.10e]" % (sol[0],sol[1],sol[2]))
#print(energy(sol))

"""
with open("AdS4QPa1_02.dat","w") as s:
    for i in range(len(sol)):
        s.write("%d %.14e \n" % (i,sol[i]))
    print("Wrote QP modes to %s" % s.name)
                        
with open("AdS4QPa1_02E.dat","w") as f:
    for i in range(len(energy(sol))):
        f.write("%d %.14e \n" % (i,energy(sol)[i]))
    print("Wrote QP mode energies to %s" % f.name)
"""    

###################################################################
###################################################################



"""
Test case of N=3
"""

def f(x):
    f = np.zeros(3)    
    
    f[0] = 2.*Tval(0) + 2.*(Rval(0,1)*(0.2**2) + Rval(0,2)*x[2]**2) + w(0,d)*x[0] \
    + 2.*(Sval(0,1,1,0)*(0.2**2) + Sval(0,2,2,0)*x[2]**2 + Sval(1,0,1,0)*(0.2**2) \
    + Sval(1,1,2,0)*x[2]*(0.2**2) + Sval(2,0,2,0)*x[2]**2)
    
    f[1] = 2.*Tval(1)*(0.2**3) + 2.*(Rval(1,0)*0.2 + Rval(1,2)*0.2*x[2]**2) \
    + w(1,d)*x[1]*0.2 + 2.*(Sval(0,1,0,1)*0.2 + Sval(0,2,1,1)*0.2*x[2] + Sval(1,0,0,1)*0.2 \
    + Sval(1,1,1,1)*(0.2**3) + Sval(1,2,2,1)*(0.2)*x[2]**2 + Sval(2,0,1,1)*0.2*x[2] \
    + Sval(2,1,2,1)*0.2*x[2]**2)
    
    f[2] = 2.*Tval(2)*x[2]**3 + 2.*(Rval(2,0) + Rval(2,1)*0.2**2)*x[2] \
    + w(2,d)*(x[0] + 2.*(x[1]-x[0]))*x[2] + 2.*(Sval(0,2,0,2)*x[2] + Sval(1,1,0,2)*(0.2**2) \
    + Sval(1,2,1,2)*(0.2**2)*x[2] + Sval(2,0,0,2)*x[2] + Sval(2,1,1,2)*(0.2**2)*x[2] \
    + Sval(2,2,2,2)*(x[2]**3))
    
    return f
    
x0 = np.ones(3)
sol = newton_krylov(f,x0)
print("Test case with explicit formulas [beta_0, beta_1, alpha_2] = [%.10e, %.10e, %.10e]" % (sol[0],sol[1],sol[2]))













