# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 19:23:12 2016

@author: bradc
"""

import numpy as np
import math
from numba import jit


R = np.genfromtxt("Mathematica_R.dat",dtype=np.float)
T = np.genfromtxt("Mathematica_T.dat",dtype=np.float)
S = np.genfromtxt("d3S_L10.dat",dtype=np.float)

N = T.shape[0]
a0 = float(1.0)
a1 = float(0.35)


d = float(3.)

def w(n,d):
    if n<0:
        return d
    else:
        return d+2.*n


@jit(nogil=True)    
def Rval(i,j):
    #print("Asked for R(%d,%d)" % (i,j))
    for row in range(R.shape[0]):
        if R[row][0]==i and R[row][1]==j:
            #print("Returning R(%d,%d) = %f" % (R[row][0],R[row][1],R[row][2]))
            return R[row][2]
    #print("Failed to find R(%d,%d)" % (i,j))


@jit(nogil=True)   
def Tval(i):
    #print("Asked for T(%d)" % i)
    for row in range(T.shape[0]):
        if T[row][0]==i:
            #print("Returning T(%d) = %f" % (T[row][0],T[row][1]))
            return T[row][1]
            
@jit(nogil=True)
def Sval(i,j,k,l):
    #print("Asked for S(%d,%d,%d,%d)" % (i,j,k,l))
    for row in range(S.shape[0]):
        if S[row][0]==i and S[row][1]==j and S[row][2]==k and S[row][3]==l:
            #print("Returning S(%d,%d,%d,%d) = %f" % (S[row][0],S[row][1],S[row][2],S[row][3],S[row][4]))
            return S[row][4]
    #print("Failed to find S(%d,%d,%d,%d)" % (i,j,k,l))


#Calculate b0 and b1 in terms of the alphas using (14) in Green et al.
def inits(a0,a1):
    mu = math.log(3./(5.*a1))
    inits = [a0,a1]
    for i in range(2,N):
        inits.append(3.*math.exp(-mu*float(i))/(2.*float(i)+3.))
    return inits

seeds = inits(a0,a1)
print(seeds)


@jit()
def makebees(seeds):
    #print("Constructing beta")
    beta = [float(0.)]
    for i in range(2):
        r = 0. ; s = 0.
        for j in range(N):
            if i!=j:
                #print("Calculating r")
                r = r + 2.*Rval(i,j)*seeds[i]*seeds[j]**2
                #print("r =",r)
                for k in range(N):
                    if k+j-i<N and k!=i and j+k>=i:
                        #print("Calculating s with (i,j,k) = (%d,%d,%d) and seeds[%d] = %f \
                        #seeds[%d] = %f, seeds[%d] = %f" %(i,j,k,j,seeds[j],k,seeds[k],j+k-i,seeds[j+k-i]))
                        s = s + 2.*Sval(j,k,j+k-i,i)*seeds[j]*seeds[k]*seeds[j+k-i]
        #print("r =",r)
        #print("s =",s)
        #print("w =",w(i,d))
        #print("seeds[%d] =" % i,seeds[i])
        #print("beta[%d] value is" % i, (-2.*Tval(i)*(seeds[i])**3 - r - s)/(w(i,d)*seeds[i]))
        beta.append((-2.*Tval(i)*(seeds[i])**3 - r - s)/(w(i,d)*seeds[i]))
    beta = beta[1:]
    
    for i in range(2,N):
        beta.append(beta[0] + float(i)*(beta[1]-beta[0]))
    #print("Done beta")
    return beta
        
b = makebees(seeds)
print("beta =",b)


@jit(nogil=True,nopython=True)
def alpha(i,x):
    if i==0:
        return a0
    if i==1:
        return a1
    if i>1:
        return x[i]

def ayes(i):
    return seeds[i]

# Use the QP equation (14) from arXiv:1507.08261
@jit(nogil=True)
def loop(b):
    F = [0.]
    for i in range(N):
        r = 0. ; s = 0.
        for j in range(N):
            if i!=j:
                r = r + 2.*Rval(i,j)*ayes(i)*ayes(j)**2
                for k in range(N):
                    if k+j-i<N and k!=i and j+k>=i:
                        s = s + 2.*Sval(j,k,j+k-i,i)*(ayes(j))*(ayes(k))*(ayes(j+k-i))
        #print("r =",r)
        #print("s =",s)
        F.append(2.*Tval(i)*(ayes(i))**3 + w(i,d)*(b[0]+float(i)*(b[1]-b[0]))*ayes(i) + r + s)
    return F[1:]
    
            
print(loop(b))            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
