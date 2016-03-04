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
N = 11
d = 3


 
"""    
Functions for solving TTF coefficients
"""
            


def b(i,d,alpha):
    # beta_0 for any N
    if i == 0:
        s = 0; r = 0
        for k in range(N):
            for l in range(N):
                if k+l<N:
                    if k == 0:
                        if l == 0:
                            s = s + 2.*Sval(l,k,k+l,0)*a0**3
                        if l == 1:
                            s = s + 2.*Sval(l,k,k+l,0)*a0*a1**2
                        if l > 1:
                            s = s + 2.*Sval(l,k,k+l,0)*a0*alpha[l]*alpha[k+l]
                    if k == 1:
                        if l == 0:
                            s = s + 2.*Sval(l,k,k+l,0)*a0*a1**2
                        if l == 1:
                            s = s + 2.*Sval(l,k,k+l,0)*alpha[k+l]*a1**2
                        if l > 1:
                            s = s + 2.*Sval(l,k,k+l,0)*a1*alpha[l]*alpha[k+l]
                    if k > 1:
                        if l == 0:
                            s = s + 2.*Sval(l,k,k+l,0)*a0*alpha[k]*alpha[k+l]
                        if l == 1:
                            s = s + 2.*Sval(l,k,k+l,0)*a1*alpha[k]*alpha[k+l]
                        if l > 1:
                            s = s + 2.*Sval(l,k,k+l,0)*alpha[l]*alpha[k]*alpha[k+l]                
            if k == 0:
                r = r + 2.*Rval(0,k)*a0**3
            if k == 1:
                r = r + 2.*Rval(0,k)*a0*a1**2
            else:
                r = r + 2.*Rval(0,k)*a0*alpha[k]
        return (-2.*Tval(0)*a0**3 -r -s)/(w(0,d)*a0)
    
    # beta_1 for any N
    if i == 1:
        s = 0; r = 0
        for k in range(N):
            for l in range(N):
                if k+l<N:
                    if k == 0:
                        if l == 0:
                            s = s + 0.
                        if l == 1:
                            s = s + 2.*Sval(l,k,k+l-1,1)*a1*a0**2
                        if l > 1:
                            s = s + 2.*Sval(l,k,k+l-1,1)*a0*alpha[l]*alpha[k+l-1]
                    if k == 1:
                        if l == 0:
                            s = s + 2.*Sval(l,k,k+l-1,1)*a1*a0**2
                        if l == 1:
                            s = s + 2.*Sval(l,k,k+l-1,1)*a1**3
                        if l > 1:
                            s = s + 2.*Sval(l,k,k+l-1,1)*a1*alpha[l]*alpha[k+l-1]
                    if k > 1:
                        if l == 0:
                            s = s + 2.*Sval(l,k,k+l-1,1)*a0*alpha[k]*alpha[k+l-1]
                        if l == 1:
                            s = s + 2.*Sval(l,k,k+l-1,1)*a1*alpha[k]*alpha[k+l-1]
                        if l > 1:
                            s = s + 2.*Sval(l,k,k+l-1,1)*alpha[l]*alpha[k]*alpha[k+l-1]                
            if k == 0:
                r = r + 2.*Rval(1,k)*a1*a0**2
            if k == 1:
                r = r + 2.*Rval(1,k)*a1**3
            else:
                r = r + 2.*Rval(1,k)*a0*alpha[k]
        return (-2.*Tval(1)*a1**3 -r -s)/(w(1,d)*a1)

    # beta_i for i !=0,1    
    if i > 1:
        return b(0,d,alpha)+(b(1,d,alpha)-b(0,d,alpha))*i*1.0


# Use the QP equation (14) from arXiv:1507.08261
def F(alpha):
    F = np.zeros_like(alpha)
    for i in range(2,N):
        s = 0
        r = 0
        F[i] = 0
        for j in range(N):
            for k in range(N):
                    if j+k-i<N:
                        #print("In S sum")
                        #print("[i,j,k] = [%d,%d,%d]" % (i,j,k))
                        s = s + 2.*Sval(j,k,j+k-i,i)*alpha[j]*alpha[k]*alpha[j+k-i]
                        #print("s =", s)
            #print("In R sum")
            #print("[i,j] = [%d,%d]" % (i,j))
            r = r + 2.*Rval(i,j)*alpha[i]*(alpha[j]**2)
            #print("r =", r)
        #print("F[%d] =" %i,F[i])
        F[i]= 2.*Tval(i)*(alpha[i]**2)*alpha[i] + r + 1.*w(i,d)*b(i,d,alpha)*alpha[i] + s
        #print("F[%d] =" %i, F[i])
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
ainits[0] = a0
ainits[1] = a1
sol = newton_krylov(F,ainits,verbose="True")
print(sol)
print(energy(sol))


with open("AdS4QPa1_02.dat","w") as s:
    for i in range(len(sol)):
        s.write("%d %.14e \n" % (i,sol[i]))
    print("Wrote QP modes to %s" % s.name)
                        
with open("AdS4QPa1_02E.dat","w") as f:
    for i in range(len(energy(sol))):
        f.write("%d %.14e \n" % (i,energy(sol)[i]))
    print("Wrote QP mode energies to %s" % f.name)
    

###################################################################
###################################################################



"""
print("X[1][0][1][0] =", Xval(1,0,1,0))
print("Y[1][0][0][1] =", Yval(1,0,0,1))
print("S[1][0][0][0] =", Sval(1,0,0,0))
print("T[1] =", Tval(2))
print("R[1][0] =", Rval(1,1))
print("b1 =", b(1,3))
print("b0 =", b(0,3))
"""

