# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:08:11 2015

@author: bradc
"""
"""
Calculate the time-averaged coefficients for a massive scalar field as described
by the recursion relations of ArXiv:1508.04943.
"""

import math
import recursiontools as rt
from scipy.special import gamma
from scipy.integrate import dblquad


# Initial values and global definitions

def w(n):
    if n<0:
        return d
    else:
        return d+2*n

def x_0(d):
    x_0 = 6*(gamma(3*d/2)*(gamma(d))**2)/(gamma(2*d)*(gamma(d/2))**3)
    return x_0
    
def y_0(d):
    y_0 = (8*gamma((3*d/2)-(1/2))*gamma((d/2)+(5/2))*(gamma(d))**2)\
    /(gamma(2*d+2)*(gamma(d/2))**4)
    return y_0
    
def W_00naught(d):
    I = dblquad(lambda x,y: ((16*(gamma(d))**2)/((gamma(d/2))**4))*((math.cos(x))**(2*d))*((math.tan(x))**(d-1))*((math.cos(y))**(2*d+1))*(math.sin(y)), \
    0,math.pi/2, lambda x:0, lambda x:x)
    return I

###############################################################################
###############################################################################

# Recursion relation for chi that takes rt.symmat(L) objects
def chi(x):
    for i in range(1,x.dim):
        for j in range(0,i+1):
            for k in range(0,j+1):
                for l in range(0,k+1):
                    x.T[i][j][k][l] = (((2*(w(i-1)+1))/(math.sqrt((i)*(i-1+d))))*(1/(2+w(i-1) \
                    + w(j) + w(k) + w(l))))*(((d-1)/2)*((w(j)**2)/(w(j)-1) + (w(k)**2)/(w(k)-1) \
                    + (w(l)**2)/(w(l)-1) - ((w(i-1)**2)/(w(i-1)**2 -1))*(1+w(j)+w(k)+w(l)))*x.getel(i-1,j,k,l) \
                    + ((w(j)*math.sqrt(j*(j+d-1)))/(w(j)-1))*x.getel(i-1,j-1,k,l) \
                    + ((w(k)*math.sqrt(k*(k+d-1)))/(w(k)-1))*x.getel(i-1,j,k-1,l) \
                    + ((w(l)*math.sqrt(l*(l+d-1)))/(w(l)-1))*x.getel(i-1,j,k,l-1) \
                    - (2+w(j)+w(k)+w(l)-w(i-1))*((math.sqrt((i-1)*(i+d-2)))/(2*(w(i-1)-1)))*x.getel(i-2,j,k,l))
    return x    

# Recursion relation for psi that takes rt.symmat(L) objects
def psi(y):
    for i in range(1,y.dim):
        for j in range(0,i+1):
            for k in range(0,j+1):
                for l in range(0,k+1):
                    y.T[i][j][k][l] = (((w(i-1)+1)/(math.sqrt((i)*(i-1+d))))*(1/(2+w(i-1) \
                    + w(j) + w(k) + w(l))))*(((d-1)/2)*((w(j)/(w(j)-1)) + (w(k)/(w(k)-1)) \
                    + (w(l)/(w(l)-1)) + 6 +(w(i-1)**2-w(j)-w(k)-w(l)+2)/(w(i-1)**2 - 1))*y.getel(i-1,j,k,l) \
                    + ((2*w(j)*math.sqrt(j*(j+d-1)))/(w(j)-1))*y.getel(i-1,j-1,k,l) \
                    + ((2*w(k)*math.sqrt(k*(k+d-1)))/(w(k)-1))*y.getel(i-1,j,k-1,l) \
                    + ((2*w(l)*math.sqrt(l*(l+d-1)))/(w(l)-1))*y.getel(i-1,j,k,l-1) \
                    - (2+w(j)+w(k)+w(l)-w(i-1))*((math.sqrt((i-1)*(i+d-2)))/(w(i-1)-1))*y.getel(i-2,j,k,l))
    return y
    
# Use chi to compute X
def makeX(x):
    for i in range(0,X.dim):
        for j in range(0,X.dim):
            for k in range(0,j+1):
                for l in range(0,k+1):
                    try:
                        X.T[i][j][k][l] = w(i)*(math.sqrt((i+1)*(i+d))*x.getel(i+1,j,k,l)/(2*(w(i)+1)) \
                        - math.sqrt(i*(i+d-1))*x.getel(i-1,j,k,l)/(2*(w(i)-1)) \
                        - (d-1)*w(i)*x.getel(i,j,k,l)/(2*(w(i)**2 - 1)))
                    except IndexError:
                        print("Index error for X.T[%d][%d][%d][%d]" % (i,j,k,l))
                        pass
    return X
    
# Use phi to compute Y
def makeY(y):
    for i in range(0,Y.dim):
        for j in range(0,Y.dim):
            for k in range(0,Y.dim):
                for l in range(0,k+1):
                    try:
                        Y.T[i][j][k][l] = w(i)*w(k)*w(l)*(math.sqrt(j*(j+d-1))*y.getel(i,j-1,k,l)/(2*(w(j)-1)) \
                        - math.sqrt((j+1)*(j+d))*y.getel(i,j+1,k,l)/(2*(w(j)+1)) \
                        -(d-1)*w(j)*y.getel(i,j,k,l)/(2*(w(j)**2 -1)))
                    except IndexError:
                        print("Index error for Y.T[%d][%d][%d][%d]" % (i,j,k,l))
                        pass
    return Y

# Use X,Y to compute S
def makeS(X,Y):
    for i in range(0,S.dim):
        for j in range(0,i+1):
            for k in range(0,j+1):
                for l in range(0,k+1):
                    if i ==k or j==k:
                        pass
                        #print("S[%d][%d][%d][%d] does not exist due to restricted sum" % (i,j,k,l))
                    else:
                        S.T[i][j][k][l] = -(1/4)*(1/(w(i)+w(j)) + 1/(w(i)-w(k)) + 1/(w(j)-w(k)))*(w(i)*w(j)*w(k)*X.getel(l,i,j,k) - w(l)*Y.getel(i,l,j,k)) \
                        -(1/4)*(1/(w(i)+w(j)) + 1/(w(i)-w(k)) - 1/(w(j)-w(k)))*(w(j)*w(k)*w(l)*X.getel(i,j,k,l) - w(i)*Y.getel(j,i,k,l)) \
                        -(1/4)*(1/(w(i)+w(j)) - 1/(w(i)-w(k)) + 1/(w(j)-w(k)))*(w(i)*w(k)*w(l)*X.getel(j,i,k,l) - w(j)*Y.getel(i,j,k,l)) \
                        -(1/4)*(1/(w(i)+w(j)) - 1/(w(i)-w(k)) - 1/(w(j)-w(k)))*(w(i)*w(j)*w(l)*X.getel(k,i,j,l) - w(i)*Y.getel(j,i,k,l))
    return S

# W_00 requires different recursion relations for k=l=0. Build these first.                    
def makeW_00_zeros():
    for i in range(1,L):
        for j in range(0,L):
            W_00.T[i][j][0][0] = (w(i-1)+1)/((w(i-1)+2)*math.sqrt((i)*(i+d-1)))*(((d-1)/2)*((w(i-1)**2 - 4)/(w(i-1)**2 -1) \
            + (w(j)**2)/(w(j)**2 - 1) - 2*(w(0)**2 + 1)/(w(0)**2 - 1))*W_00.getel2(i-1,j,0,0) + (w(i-1)-2)*math.sqrt((i-1)*(i+d-2))*W_00.getel2(i-2,j,0,0)/(w(i-1)-1))
            W_00.T[j][i][0][0] = W_00.T[i][j][0][0]
    return W_00
# Then use regular recursion relation for W_00[i][j][k][l] when k != l, different recursion
# relation when k = l, which uses the result of makeW_00_zeros().         
def makeW_00():
    for i in range(0,W_00.dim):
        for j in range(0,W_00.dim):
            for k in range(0,W_00.dim):
                for l in range(0,k+1):
                    if k == l:
                        try:
                            W_00.T[i][j][l+1][l+1] = ((w(l)+1)/(w(l+1)-1))*W_00.getel2(i,j,l,l)
                        except IndexError:
#                            print("Index error for W.T[%d][%d][%d][%d]" % (i,j,l+1,l+1))
                            pass
                    else:
                        try:
                            W_00.T[i][j][k][l] = (X.getel3(l,i,j,k) - X.getel3(k,i,j,l))/(w(k)**2 - w(l)**2)
                        except IndexError:
                            print("Index error for W.T[%d][%d][%d][%d]" % (i,j,k,l))
                            pass
    return W_00

def makeW_10():
    for i in range(0,W_10.dim):
        for j in range(0,W_10.dim):
            for k in range(0,W_10.dim):
                for l in range(0,k+1):
                    if l == k:
                        W_10.T[i][j][k][l] = (1/2)*((w(i)**2 + w(j)**2 -4)*W_00.getel2(i,j,k,l) - 2*(d-1)*x.getel(i,j,k,l) - 2*X.getel3(i,j,k,l) \
                        -2*X.getel3(j,i,k,l) - X.getel3(k,i,j,l) - X.getel3(l,i,j,k))
                    else:
                        W_10.T[i][j][k][l] = (Y.getel2(i,k,j,l) - Y.getel2(i,l,j,k))/(w(k)**2 - w(l)**2)
    return W_10                    
    
def makeT():
    T = [None]*L
    for i in range(0,L):
        T[i] = (w(i)**2)*X.getel3(i,i,i,i)/2 + 3*Y.getel2(i,i,i,i)/2 + 2*(w(i)**4)*W_00.getel2(i,i,i,i) + 2*(w(i)**2)*W_10.getel2(i,i,i,i)
    return T
    
def makeR():
    for i in range(0,R.dim):
        for j in range(0,R.dim):
            if i == j:
                pass
                print("R[%d][%d] does not exist due to restricted sum" % (i,j))
            else:
                R.B[i][j] = ((w(i)**2 + w(j)**2)/(w(j)**2 - w(i)**2))*((w(j)**2)*X.getel3(i,j,j,i) - (w(i)**2)*X.getel3(j,i,i,j))/2 \
                + 2*((w(j)**2)*Y.getel2(i,j,i,j) - (w(i)**2)*Y.getel2(j,i,j,i))/(w(j)**2 - w(i)**2) \
                + (Y.getel2(i,i,j,j) + Y.getel2(j,j,i,i))/2 \
                +(w(i)**2)*(w(j)**2)*(X.getel3(i,j,j,i) - X.getel3(j,i,j,i))/(w(j)**2 - w(i)**2) \
                + (w(i)**2)*(w(j)**2)*(W_00.getel2(j,j,i,i) + W_00.getel2(i,i,j,j)) \
                + (w(i)**2)*(W_10.getel2(j,j,i,i)) + (w(j)**2)*(W_10.getel2(i,i,j,j))
    return R

# Create output files of entries for X,Y,R,T listed by index value with 14 significant figures    
def outputs(X,Y,R,T):
    print("Caculating X,Y,R,T from the recursion relations using L=%d and d=%d" % (L,d))
    with open("d3T.dat","w") as t:
        for i in range(0,len(T)):
            t.write("%d %.14e \n" %(i,T[i]))
        print("Wrote T to %s" % t.name)
    with open("d3R.dat","w") as r:
        for i in range(0,R.dim):
            for j in range(0,i+1):
                try:
                    r.write("%d %d %.14e \n" % (i,j,R.B[i][j]))
                except TypeError:
                    r.write("%d %d None \n" % (i,j))
        print("Wrote R to %s" % r.name)
    with open("d3X.dat","w") as x:
        for i in range(0,X.dim):
            for j in range(0,i+1):
                for k in range(0,j+1):
                    for l in range(0,k+1):
                        x.write("%d %d %d %d %.14e \n" % (i,j,k,l,X.T[i][j][k][l]))
        print("Wrote X to %s" % x.name)
    with open("d3Y.dat","w") as y:
        for i in range(0,Y.dim):
            for j in range(0,i+1):
                for k in range(0,j+1):
                    for l in range(0,k+1):
                        y.write("%d %d %d %d %.14e \n" % (i,j,k,l,Y.T[i][j][k][l]))
        print("Wrote Y to %s" % y.name)
    print("Done")
                    
######################################################################
######################################################################
     
"""
Maximum "level" to be calculated, L (non-inclusive), and number of dimensions, d
"""
L=2
d=3

"""
Chi and psi must be calculated to level L+1
"""
x = rt.symmat(L+1)
x.build()
x.T[0][0][0][0] = x_0(d)
chi(x)
print("x.T =", x.T,"\n")
y = rt.symmat(L+1)
y.build()
y.T[0][0][0][0] = y_0(d)
psi(y)
#print("y.T =", y.T, "\n")

"""
Using chi and psi, X and Y are computed to level L
"""
X = rt.symmat(L)
X.build3()
makeX(x)
print("X =", X.T,"\n")
Y = rt.symmat(L)
Y.build2()
makeY(y)
print("Y =", Y.T,"\n")

"""
S is computed to level L using X and Y. Note that values prohibited by the restricted
sums from equation 7 in ArXiv:1508.04943 will remain as "None" in S 
"""
S = rt.symmat(L)
S.build()
makeS(X,Y)
#print("S =", S.T,"\n")

"""
Both R and T require calculating W_00 and W_10 first
"""
# W_00 is computed to level L
W_00 = rt.symmat(L)
W_00.build2()
W_00.T[0][0][0][0] = W_00naught(d)[0]
makeW_00_zeros()
makeW_00()
print("W_00 =", W_00.T, "\n") 
# W_10 is computed to level L
W_10 = rt.symmat(L)
W_10.build2()
makeW_10()
print("W_10 =", W_10.T, "\n")

"""
Now use the results of W_00 and W_10 to calculate R and T
"""
# T is computed to level L
T = makeT()
print("T =", T, "\n")
# R is computed to level L
R = rt.symmat(L)
R.buildR()
makeR()
print("R =", R.B, "\n") 

"""
Finally, output the results into individual files
"""
#outputs(X,Y,R,T)

print(x.getel(0,0,0,0))
print(x.getel(0,0,1,0))
print(x.getel(1,0,1,0))
    
    
    
    


