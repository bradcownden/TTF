# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:31:20 2016

@author: bradc
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

###################################################################
###################################################################


S = np.genfromtxt("d3S_L10.dat")
X = np.genfromtxt("d3X_L10.dat")
Y = np.genfromtxt("d3X_L10.dat")


###################################################################
###################################################################
  
def w(n,d):
    if n<0:
        return float(d)
    else:
        return d+2.*n
     
def ks(i,d):
    return 2.*math.sqrt(math.factorial(i)*math.factorial(i+d-1))/gamma(i + 0.5*d)
    
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
    
def ep(i,d):
    f = diff(ks(i,d)*((symcos(t))**d)*jacobi(i,0.5*d -1,0.5*d,symcos(2*t)),t)
    return lambdify(t,f)
    
def A(i,j,d):
    return quad(lambda x: (ks(i,d)*((math.cos(x))**d)*eval_jacobi(i,0.5*d -1,0.5*d,math.cos(2*x))) \
    * (ks(j,d)*((math.cos(x))**d)*eval_jacobi(j,0.5*d -1,0.5*d,math.cos(2*x))) * math.sin(x)*math.cos(x), \
    0, math.pi/2.)[0]
    
def V(i,j,d):
    return quad(lambda x: (ep(i,d)(x))*(ep(j,d)(x))*math.sin(x)*math.cos(x),0,math.pi/2.)[0]
    
def T(i,d):
    return ((w(i,d)**2)*(X[i][1])/2. + 3*(Y[i][1])/2. + 2*(w(i,d)**4)*W00(i,i,i,i,d) + 2*(w(i,d)**2)*W10(i,i,i,i,d) \
    - (w(i,d)**2)*(A(i,i,d) + (w(i,d)**2)*V(i,i,d)))
        
print(T(1,3))

    
















###################################################################
###################################################################