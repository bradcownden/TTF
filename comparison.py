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
ind = np.lexsort((Xme[:,4],Xme[:,3],Xme[:,2],Xme[:,1]))
Xmeprime = Xme[ind]

Xnils = np.loadtxt("NilsData/X_30.dat",ndmin=2)
Xnils.sort(axis=1)

comp = np.zeros(Xme.shape[0])

for i in range(Xmeprime.shape[0]):
    comp[i] = (Xmeprime[i][0] - Xnils[i][0])*100/Xmeprime[i][0]
    
print(comp[0])
print(comp[1])
    

# Plot the results

x = range(Xmeprime.shape[0])
mp.plot(x,comp,marker=".")
mp.ylabel('Percent Difference in X Coefficients')
mp.xlabel('Index Number')
mp.title('X Coefficients Comparison')
mp.show()
    
