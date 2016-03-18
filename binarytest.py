# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 08:21:55 2016

@author: bradc
"""

import numpy as np

Ttype = np.dtype([('',np.int32),('',np.float)])
Rtype = np.dtype([('',np.int32),('',np.int32),('',np.float)])
Stype = np.dtype([('',np.int32),('',np.int32),('',np.int32),('',np.int32),('',np.float)])

Rbin = np.fromfile("/Users/bradc/Dropbox/AdS_CFT/MasslessScalarRecursion/AdS4_R_j100.bin",Rtype)
Tbin = np.fromfile("/Users/bradc/Dropbox/AdS_CFT/MasslessScalarRecursion/AdS4_T_j100.bin",Ttype)
Sbin = np.fromfile("/Users/bradc/Dropbox/AdS_CFT/MasslessScalarRecursion/AdS4_S_j100.bin",Stype)

smallR = Rbin[0:5]
print(smallR,smallR.shape,smallR[1][0])
sRsort = np.sort(smallR)
sRten = np.zeros((1,3))
for i in range(smallR.shape[0]):
    sRten = np.vstack((sRten,np.array([smallR[i][0],smallR[i][1],smallR[i][2]])))

print(sRten)


# convert to numpy array
T = np.sort(Tbin)
Tten = np.zeros((1,2))
for i in range(Tbin.shape[0]):
    Tten = np.vstack((Tten,np.array([T[i][0],T[i][1]])))
T = Tten[1:]
print(T[0:10])
    
R = np.sort(Rbin)
Rten = np.zeros((1,3))
for i in range(Rbin.shape[0]):
    Rten = np.vstack((Rten,np.array([R[i][0],R[i][1],R[i][2]])))
R = Rten[1:]
print(R)

S = np.sort(Sbin)
Sten = np.zeros((1,5))
for i in range(Sbin.shape[0]):
    Sten = np.vstack((Sten,np.array([S[i][0],S[i][1],S[i][2],S[i][3],S[i][4]])))
S = Sten[1:]
print(S)














