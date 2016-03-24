# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:45:18 2016

@author: bradc

Test the residual of newton_krylov optimzation 
"""

import numpy as np
from numba import jit

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

data = np.genfromtxt("./data/AdS4QP_j50_a021.dat",dtype=np.float)

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
                if np.isnan(S[row][4]) == True:
                    return 0
                else:
                    return S[row][4]
        #print("S[%d][%d][%d][%d] not found" % (i,j,k,l))
        return 0
@jit(nopython=True)      
def Rval(i,j):
    for row in range(R.shape[0]):
        if R[row][0]==i and R[row][1]==j:
            if np.isnan(R[row][2])==True:
                return 0
            else:
                return R[row][2]
    #   print("R[%d,%d] not found" % (i,j))    
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
a0 = np.float(1.0)
a1 = np.float(0.20)
N = T.shape[0]
d = 3.
b0 = np.float(5.25276336e+01)
b1 = np.float(1.23133803e+02)

 
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
    if i<0:
        #print("Index error in alpha, called for i=%d" % i)
        return 0



# Use the QP equation (14) from arXiv:1507.08261
@jit()
def residual(dat):
    F = np.zeros(data.shape[0])
    for i in range(N):
        s = 0; r = 0
        for j in range(N):
            for k in range(N):
                if j+k-i<N:
                    s = s +2.*Sval(j,k,j+k-i,i)*dat(j)*dat(k)*dat(j+k-i) 
            r = r + 2.*Rval(i,j)*dat(i)*dat(j)**2
        F[i] = 2.*Tval(i)*dat(i)**3 + w(i,d)*(b0+np.float(i)*(b1-b0))*dat(i) + r + s
        print("Completed %d/%d" % (i+1,N))
        print("F[%d] =" % i, F[i])
    return F

@jit()
def check():
    return residual(dat)

res = check()

with open("./data/residuals_j50_a021.dat","w") as f:
    for i in range(len(res)):
        f.write("%d %.14e \n" % (i,res[i]))
    print("Wrote residuals to %s" % f.name)



























############################################################
############################################################