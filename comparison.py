# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:44:31 2015

@author: bradc
"""

"""
Compare values of X, Y, S, R, T between those calculated by recursion
relation vs those calculated by integration. Read in data from files, ensure
sorting convention is the same, then calculate percent difference
"""

import numpy as np
import matplotlib.pyplot as mp


# Load the data files and calculate the differences

Xme = np.loadtxt("d3X.dat",ndmin=2)
Xme.sort(axis=1)
indx = np.lexsort((Xme[:,4],Xme[:,3],Xme[:,2],Xme[:,1]))
Xmeprime = Xme[indx]

Xnils = np.loadtxt("NilsData/X_30.dat",ndmin=2)
Xnils.sort(axis=1)

compx = np.zeros(Xmeprime.shape[0])

for i in range(Xmeprime.shape[0]):
    compx[i] = (Xmeprime[i][0] - Xnils[i][0])

###########################################################
    
Yme = np.loadtxt("d3Y.dat",ndmin=2)
Yme.sort(axis=1)
indy = np.lexsort((Yme[:,4],Yme[:,3],Yme[:,2],Yme[:,1]))
Ymeprime = Yme[indy]

Ynils = np.loadtxt("NilsData/Y_30.dat",ndmin=2)
Ynils.sort(axis=1)

compy = np.zeros(Ymeprime.shape[0])

for i in range(Ymeprime.shape[0]):
    compy[i] = (Ymeprime[i][0] - Ynils[i][0])
        
    
###########################################################
###########################################################    

# Plot the results

mp.figure(1)

mp.subplot(211)
x = range(Xmeprime.shape[0])
mp.plot(x,compx,'bo')
mp.ylabel('Difference in X Coefficients')
mp.xlabel('Index Number')
mp.title('X Coefficients Comparison')

mp.subplot(212)
x = range(Ymeprime.shape[0])
mp.plot(x,compy,'ro')
mp.ylabel('Difference in Y Coefficients')
mp.xlabel('Index Number')
mp.title('Y Coefficients Comparison')
mp.show()
    
