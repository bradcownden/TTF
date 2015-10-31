# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:52:21 2015

@author: bradc
"""
"""
Create integral expressions for TTF coefficients to check against
recursion relations
"""

import recursiontools as rt
from sympy import Derivative, symbols, Function, Symbol, sqrt, cos, factorial
import scipy.integrate
from sympy.mpmath import jacobi, gamma
from scipy.special import gamma
from sympy.abc import n,x
import math

############################################################
############################################################

def w(n):
    w = d + 2*n
    return w

def x_0(d):
    x_0 = 6*(gamma(3*d/2)*(gamma(d))**2)/(gamma(2*d)*(gamma(d/2))**3)
    return x_0
    
def y_0(d):
    y_0 = (8*gamma((3*d/2)-(1/2))*gamma((d/2)+(5/2))*(gamma(d))**2)\
    /(gamma(2*d+2)*(gamma(d/2))**4)
    return y_0
    
def kay(n):
    return lambda x: 2*math.sqrt(math.factorial(n)*math.factorial(n+d-1))/gamma(n+(d/2))    
    
def e(n,x):
    J = lambda x: jacobi(n,d/2-1,d/2,math.cos(2*x))
    return lambda x: kay(n)*((math.cos(x))**d)*J
    
def makeX(X):
    J = jacobi
    for i in range(0,X.dim):
        for j in range(0,i+1):
            for k in range(0,j+1):
                for l in range(0,k+1):
                    print("In caculation loop")
                    eiprime = lambda x: Derivative((math.cos(x))**d*J(i,d/2-1,d/2,math.cos(2*x)),x)                  
                    ej = lambda x: (math.cos(x))**d*jacobi(j,d/2-1,d/2,math.cos(2*x))
                    ek = lambda x: (math.cos(x))**d*jacobi(k,d/2-1,d/2,math.cos(2*x))
                    el = lambda x: (math.cos(x))**d*jacobi(l,d/2-1,d/2,math.cos(2*x))
                    X.T[i][j][k][l] = scipy.integrate.quad(eiprime,0,math.pi/2)[0]
    return X
    

############################################################
############################################################

L=2
d=3
X = rt.symmat(L)
X.build()
print("X =", X.T)
f = symbols('f',cls=Function)
x = Symbol('x')
n = Symbol('n')
f = Derivative(2*sqrt(factorial(n)*factorial(n+d-1))*jacobi(n,d/2-1,d/2,cos(2*x))*((cos(x))**d),x,evaluate=True)
print(f)