# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:46:14 2015

@author: bradc
"""

"""
Calculation of W_00 and W_10 values based off the explicit forms given in
ArXiv:1507.08261 for d=3
"""

from scipy.special import psi
import math
import numpy as np

####################################################################
####################################################################

L = 2
W_00vals = np.zeros((L+1,5))
W_10vals = np.zeros((L+1,5))

####################################################################
####################################################################

def dd(x):
    if x == 0:
        return 1
    else:
        return 0

def W_00(m,l):
    val = dd(m-l)*(-(2*(m**2)*(2*l+3) + 4*m*(l**2 -2) + 2*l*(3*l+5) + 3))/(16*math.pi*(m+1)*(m+2)*(l+1)*(l+2)*(2*l+3)) \
    - np.sign(m-l)*(((2*m +3)**2)*(2*l*(l+3) +5))/(16*math.pi*(m+1)*(m+2)*(l+1)*(l+2)*(2*l+3)) \
    + ((2*m +3)**2)*(-psi(m+1)+psi(m+3/2)+2*math.log(2))/(4*math.pi*(m+1)*(m+2)) \
    -(8*(m**4)*(l+1)*(2*l*(l+4)+7)+ 8*(m**3)*(l+1)*(l*(14*l+55)+48) + (m**2)*(4*l*(l*(73*l+355)+527)+979) \
    + m*(4*l*(17*l*(15*l+24)+602)+1113) + 2*(l*(l*(74*l+351)+515)+237))/(16*math.pi*((m+1)**2)*((m+2)**2)*(l+1)*(l+2)*(2*l+3))
    return val
    
def W_10(m,n):
    val = -dd(m-n)*((2*m+3)**2)*(4*(m**4)*(2*n+3) - 8*(m**3)*(n*(n-3)-7) -2*(m**2)*(2*n*(6*n*(n+6) +41) -3) \
    + 4*m*(n*(n*(6*n*(n+3) -13) -54) -18) + 3*(2*n*(n*(6*(n**2) +28*n+41) +21) +9))/(16*math.pi*(m+1)*(m+2)*(n+1)*(n+2)*(2*n+3)) \
    - dd(m-n-1)*(m+2)*((2*m+3)**2)*(n+1)*((-m+n+1)**2)/(4*math.pi*(m+1)*(n+2)*(2*n+3)) \
    + dd(m-n+1)*(m+1)*((2*m+3)**2)*(n+2)*((m-n+1)**2)/(math.pi*(m+2)*(n+1)*(8*n+12)) \
    - np.sign(m-n)*((2*m+3)**2)*(4*(m**2)*(2*n*(n+3) +5) +12*m*(2*n*(n+3) +5) -2*n*(n+3)*(8*n*(n+3) +27) -37)/(16*math.pi*(m+1)*(m+2)*(n+1)*(n+2)*(2*n+3)) \
    - (2*m+3)*(16*(m**5)*(n+1)*(2*n*(n+4) +7) + 8*(m**4)*(n+1)*(2*n*(17*n +67) +117) \
    + (m**3)*(8*n*(n*(n*(93-4*n) +517) +797) +3018) + (m**2)*(4*n*(n*(n*(187-36*n) + 1458) +2408) +4701) \
    + m*(4*n*(n*(n*(31-52*n) + 936) +1736) +3545) + 2*(n*(n*(423-2*n*(24*n+31)) +947) + 513))/(16*math.pi*((m+1)**2)*((m+2)**2)*(n+1)*(n+2)*(2*n+3))
    return val
    


####################################################################
####################################################################
    
def makeW_00(L):
    for i in range(L):
        for j in range(L):
            print("i = %d, j = %d" % (i,j))
            W_00vals[i][0] = W_00vals[i][1] = i
            W_00vals[i][2] = W_00vals[i][3] = j
            W_00vals[i][4] = W_00(i,j)
    return W_00vals
            
            
def makeW_10(L):
    for i in range(L):
        for j in range(L):
            W_10vals[i][0] = W_10vals[i][1] = i
            W_10vals[i][2] = W_10vals[i][3] = j
            W_10vals[i][4] = W_10(i,j)
    return W_10vals


####################################################################
####################################################################

def main():
    makeW_00(L)
    print(W_00vals)
    print("W_00(%d,%d,%d,%d) =" % (0,0,0,0), W_00(0,0), "\n")
#    makeW_10(L)
    print("W_10(%d,%d,%d,%d) =" % (0,0,0,0), W_10(0,0), "\n")
    

main()

####################################################################
####################################################################
