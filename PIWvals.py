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

L = 5
W_00vals = np.zeros((L**2,5))
W_10vals = np.zeros((L**2,5))
Vvals = np.zeros((L**2,3))
Avals = np.zeros((L**2,3))

####################################################################
####################################################################

def dd(x):
    if x == 0:
        return 1
    else:
        return 0

def W_00(m,l):
    val = dd(m-l)*(-2.*(m**2)*(2.*l+3.) + 4.*m*(l**2 -2) + 2*l*(3*l+5) + 3)/(16.*math.pi*(m+1)*(m+2)*(l+1)*(l+2)*(2*l+3)) \
    - np.sign(m-l)*(((2.*m +3.)**2)*(2*l*(l+3) +5))/(16*math.pi*(m+1)*(m+2)*(l+1)*(l+2)*(2*l+3)) \
    + ((2*m +3)**2)*(-psi(m+1)+psi(m+1.5)+2*math.log(2))/(4*math.pi*(m+1)*(m+2)) \
    -(8*(m**4)*(l+1)*(2*l*(l+4)+7)+ 8*(m**3)*(l+1)*(l*(14*l+55)+48) + (m**2)*(4*l*(l*(73*l+355)+527)+979) \
    + m*(4*l*(17*l*(5*l+24)+602)+1113) + 2*(l*(l*(74*l+351)+515)+237))/(16*math.pi*((m+1)**2)*((m+2)**2)*(l+1)*(l+2)*(2*l+3))
    return val

def W_10(m,n):
    val = -dd(m-n)*((2*m +3)**2)*(4*(m**4)*(2*n+3) - 8*(m**3)*(n*(n-3) -7) -2*(m**2)*(2*n*(6*n*(n+6) +41) -3) \
    + 4*m*(n*(n*(6*n*(n+3) -13) -54) -18) + 3*(2*n*(n*(6*(n**2)+28*n+41) +21) +9))/(16*math.pi*(m+1)*(m+2)*(n+1)*(n+2)*(2*n+3)) \
    - dd(m-n-1)*(m+2)*((2*m+3)**2)*(n+1)*((-m+n+1)**2)/(4*math.pi*(m+1)*(n+2)*(2*n+3)) \
    + dd(m-n+1)*(m+1)*((2*m+3)**2)*(n+2)*((m-n+1)**2)/(math.pi*(m+2)*(n+1)*(8*n+12)) \
    - np.sign(m-n)*((2*m+3)**2)*(4*(m**2)*(2*n*(n+3) +5) +12*m*(2*n*(n+3) +5) -2*n*(n+3)*(8*n*(n+3) +27) -37)/(16*math.pi*(m+1)*(m+2)*(n+1)*(n+2)*(2*n+3)) \
    - (2*m+3)*(16*(m**5)*(n+1)*(2*n*(n+4) +7) +8*(m**4)*(n+1)*(2*n*(17*n +67) +117) + (m**3)*(8*n*(n*(n*(93-4*n) +517) +797) +3018) \
    + (m**2)*(4*n*(n*(n*(187-36*n) +1458) +2408) +4701) + m*(4*n*(n*(n*(31-52*n) +936) +1736) +3545) \
    + 2*(n*(n*(423-2*n*(24*n+31)) +947) +513))/(16*math.pi*((m+1)**2)*((m+2)**2)*(n+1)*(n+2)*(2*n+3))
    return val

def V(i,j):
    if (i+j)%2 == 0:
        val = ((2*i+3)*(2*j+3)*(psi((i+j+1)/2.) - psi((i-j+1)/2.)) - ((2*i+3)**2)/(i+j+3) + i*(i+2)/(i-j-1) \
        + (i+1)*(i+3)/(-i+j-1) + (6-8*i*(i+1))/(i+j+1) + 4*(3*i+4))/(2*math.pi*math.sqrt((i+1)*(i+2)*(j+1)*(j+2)))
    else:
        val = (-8*(j**4)*(5*i+7) -4*(j**3)*(i*(14*i + 85) + 93) + 8*(j**2)*(i*(i*(i-21) -93) -89) + 4*j*(i*(i*(i*(6*i+37) +19) -102) -108) \
        -2*(2*i+3)*(2*j+3)*(i-j)*(i+j)*(i+j+2)*(i+j+4)*(psi((i-j)/2.) - psi((i+j)/2.)) + 16*i*(i+1)*(2*i*(i+5) +9))/(4*math.pi*(i-j)*(i+j)*(i+j+2)*(i+j+4)*math.sqrt((i+1)*(i+2)*(j+1)*(j+2)))
    return val
    
def A(i,j):
    if (i+j)%2 == 0:
        val = (2*i+3)*(2*j+3)*((2*i*(i+3) + 2*j*(j+3) +7)*(psi((i+j+3)/2.) - psi((i-j+1)/2.)) \
        -2*(i+1)*(j+1)*(-2*(j**2)*(i-4) -2*i*j*(i+8) + i*(2*i*(i+4) -3) +2*(j**3) -3*j -13)/((i+j+3)*((i-j)**2 -1)))/(2*math.pi*math.sqrt((i+1)*(i+2)*(j+1)*(j+2)))
    else:
        val = (2*i+3)*(2*j+3)*((2*i*(i+3) +2*j*(j+3) +7)*(psi((i+j+2)/2.) - psi((i-j)/2.)) \
        - 4*i*j -(4*i*(i+3) + 7)/(i-j) + (7*i*(i+2) +6)/(i+j+2) + (i+1)*(i+3)/(i+j+4) -8*i)/(2*math.pi*math.sqrt((i+1)*(i+2)*(j+1)*(j+2)))
    return val

####################################################################
####################################################################
    
def makeW_00(L):
    row=0
    for i in range(L):
        for j in range(L):
            W_00vals[row][0] = W_00vals[row][1] = i
            W_00vals[row][2] = W_00vals[row][3] = j
            W_00vals[row][4] = W_00(i,j)
            
            row=row+1
    return W_00vals
            
            
def makeW_10(L):
    row=0
    for i in range(L):
        for j in range(L):
            W_10vals[row][0] = W_10vals[row][1] = i
            W_10vals[row][2] = W_10vals[row][3] = j
            W_10vals[row][4] = W_10(i,j)
            
            row=row+1
    return W_10vals
    
    
def makeV(L):
    row=0
    for i in range(L):
        for j in range(0,i+1):
            Vvals[row][0] = i
            Vvals[row][1] = j
            Vvals[row][2] = V(i,j)
            
            row=row+1
    return Vvals


def makeA(L):
    row=0
    for i in range(L):
        for j in range(0,i+1):
            Avals[row][0] = i
            Avals[row][1] = j
            Avals[row][2] = A(i,j)
            
            row=row+1
    return Avals


    
def outputs():
    print("Writing outputs for W_00 and W_10 up to L = %d: \n" % L)
    
    with open("PIW_00.dat","w") as f1:
        for row in range(W_00vals.shape[0]):
            f1.write("%d %d %d %d %.14e \n" % (W_00vals[row][0],W_00vals[row][1], \
            W_00vals[row][2],W_00vals[row][3],W_00vals[row][4]))
        print("Wrote W_00 to %s" % f1.name)
        
    with open("PIW_10.dat","w") as f2:
        for row in range(W_10vals.shape[0]):
            f2.write("%d %d %d %d %.14e \n" % (W_10vals[row][0],W_10vals[row][1], \
            W_10vals[row][2],W_10vals[row][3],W_10vals[row][4]))
        print("Wrote W_10 to %s" % f2.name)
        
    with open("PIV.dat","w") as f3:
        for row in range(Vvals.shape[0]):
            if Vvals[row][0] == Vvals[row][2]:
                break
            else:
                f3.write("%d %d %.14e \n" % (Vvals[row][0],Vvals[row][1],Vvals[row][2]))
        print("Wrote V to %s" % f3.name)
        
    with open("PIA.dat","w") as f4:
        for row in range(Avals.shape[0]):
            if Avals[row][0] == Avals[row][2]:
                break
            else:
                f4.write("%d %d %.14e \n" % (Avals[row][0],Avals[row][1],Avals[row][2]))
        print("Wrote A to %s" % f4.name)


####################################################################
####################################################################

def main():
    makeW_00(L)
    print("W_00 =", W_00vals, "\n")
    makeW_10(L)
    print("W_10 =", W_10vals, "\n")
    makeV(L)
    print("V =", Vvals, "\n")
    makeA(L)
    print("A =", Avals, "\n")
    

    

main()
outputs()

####################################################################
####################################################################
