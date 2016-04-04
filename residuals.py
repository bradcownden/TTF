# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:45:18 2016

@author: bradc

Test the residual of newton_krylov optimzation 
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

############################################################
############################################################

"""
Read in S,R,T values from binary files
"""

Ttype = np.dtype([('',np.int32),('',np.float)])
Rtype = np.dtype([('',np.int32),('',np.int32),('',np.float)])
Stype = np.dtype([('',np.int32),('',np.int32),('',np.int32),('',np.int32),('',np.float)])

Rbin = np.fromfile("AdS4_R_j50.bin",Rtype)
Tbin = np.fromfile("AdS4_T_j50.bin",Ttype)
Sbin = np.fromfile("AdS4_S_j50.bin",Stype)

"""
Convert binary data to sorted numpy arrays
"""
@jit()
def Tsort(Tbin):
    T = np.sort(Tbin)
    Tten = np.zeros((1,2))
    for i in range(Tbin.shape[0]):
        Tten = np.vstack((Tten,np.array([T[i][0],T[i][1]])))
    return Tten[1:]
    

@jit()
def Rsort(Rbin):
    R = np.sort(Rbin)
    Rten = np.zeros((1,3))
    for i in range(Rbin.shape[0]):
        Rten = np.vstack((Rten,np.array([R[i][0],R[i][1],R[i][2]])))
    return Rten[1:]


@jit()
def Ssort(Sbin):
    S = np.sort(Sbin)
    Sten = np.zeros((1,5))
    for i in range(Sbin.shape[0]):
        Sten = np.vstack((Sten,np.array([S[i][0],S[i][1],S[i][2],S[i][3],S[i][4]])))
    return Sten[1:]


T = Tsort(Tbin)
R = Rsort(Rbin)
S = Ssort(Sbin)

#S = np.genfromtxt("d3S_L10.dat",dtype=np.float)
#R = np.genfromtxt("Mathematica_R.dat",dtype=np.float)
#T = np.genfromtxt("Mathematica_T.dat",dtype=np.float)

"""
Read in results of optimization
"""

data = np.genfromtxt("./data/AdS4QP_j50_a035.dat",dtype=np.float)
nils = np.genfromtxt("./NilsData/betas/QpAdS4j50r0a3.5000e-01.txt",dtype=np.float)

print("Data read-in complete")

############################################################
############################################################


"""
Access tensor elements
"""
     
@jit(nopython=True)    
def Sval(i,j,k,l):
    if i<0 or j<0 or k<0 or l<0:
        return 0
    else:
        for row in range(S.shape[0]):
            if S[row][0]==i and S[row][1]==j and S[row][2]==k and S[row][3]==l:
                return S[row][4]
        return 0
        
@jit(nopython=True)      
def Rval(i,j):
    for row in range(R.shape[0]):
        if R[row][0]==i and R[row][1]==j:
            return R[row][2]    
    return 0

@jit(nopython=True)    
def Tval(i):
    for row in range(T.shape[0]):
        if T[row][0]==i:
            return T[row][1]

@jit()
def w(n,d):
    if n<0:
        return np.float(d)
    else:
        return np.float(d)+2.*n

############################################################
############################################################


"""
Inputs for the QP mode solver
"""

# Initial values for alpha_0 and alpha_1; maximum number N = j_max; dimension d
a0 = data[0][1]
a1 = data[1][1]
N = T.shape[0]
d = 3.
b0 = data[data.shape[0]-2][1]
b1 = data[data.shape[0]-1][1]
b0nils = nils[0][1]
b1nils = nils[1][1]

nils = nils[2:nils.shape[0]-1]
data = data[:data.shape[0]-2]

 
"""    
Functions for solving for QP coefficients
"""

# Function that returns the input values a0, a1 when i=0,1 and returns the variable
# being optimized when i>1
@jit(nopython=True)
def dat(i):
    if i==0:
        return a0
    if i==1:
        return a1
    if i>1:
        return data[i][1]
    

@jit(nopython=True)
def nilsdat(i):
    if i==0:
        return nils[0][1]
    if i==1:
        return nils[1][1]
    if i>1:
        return nils[i][1]
   

# Use the QP equation (14) from arXiv:1507.08261
@jit()
def residual(dat):
    print("Calculating residuals for 'data'")
    F = np.zeros_like(data,dtype=np.float)
    for i in range(N):
        s = 0; r = 0
        for j in range(N):
            if i!=j:
                for k in range(N):
                    if j+k-i<N and i<j+k and i!=k:
                        s = s +2.*Sval(j,k,j+k-i,i)*dat(j)*dat(k)*dat(j+k-i) 
                r = r + 2.*Rval(i,j)*dat(i)*dat(j)**2
        F[i] = 2.*Tval(i)*dat(i)**3 + w(i,d)*(b0+np.float(i)*(b1-b0))*dat(i) + r + s
        print("F[%d] =" % i, F[i])
    return F


@jit()
def nilsresidual(nilsdat):
    print("Calculating residuals for 'nils'")
    F = np.zeros_like(nils,dtype=np.float)
    for i in range(N):
        s = 0; r = 0
        for j in range(N):
            if i!=j:
                for k in range(N):
                    if j+k-i<N and i<j+k and i!=k:
                        s = s +2.*Sval(j,k,j+k-i,i)*nilsdat(j)*nilsdat(k)*nilsdat(j+k-i) 
                r = r + 2.*Rval(i,j)*nilsdat(i)*nilsdat(j)**2
        F[i] = 2.*Tval(i)*nilsdat(i)**3 + w(i,d)*(b0nils+np.float(i)*(b1nils-b0nils))*nilsdat(i) + r + s
        print("F[%d] =" % i, F[i])
    return F


############################################################
############################################################

"""
Run the comparison, plot and write the results
"""

@jit()
def check():
    return residual(dat)
    
@jit()
def nilscheck():
    return nilsresidual(nilsdat)

res = check()
nilsres = nilscheck()

"""
with open("./data/residuals_j50_a040.dat","w") as f:
    for i in range(len(res)):
        f.write("%d %.14e \n" % (i,res[i]))
    print("Wrote residuals to %s" % f.name)
"""
@jit()
def plots():
    plt.plot(res,'.-g',label='Brad') 
    plt.plot(nilsres,'.-b',label='Nils') 
    plt.xlabel('')
    plt.ylabel('Optimization Residuals')
    plt.title('Testing Residuals From Nils & Brad\'s Optimizations: a1=.35')
    plt.legend(loc=4)
    plt.grid(True)
    plt.show()
    
plots()






























############################################################
############################################################