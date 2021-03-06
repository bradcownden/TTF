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
        return float(d)
    else:
        return d+2.*n

def x_0(d):
    x_0 = 6.*(gamma(3*d/2.))*((gamma(d))**2)/(gamma(2.*d)*(gamma(d/2.))**3)
    return x_0
    
def y_0(d):
    y_0 = ((2**(2*d -2))*(2+d)*gamma((3*d/2.) - 1)*(gamma(d/2. - 1/2.))**2)/(math.pi*gamma(2.*d)*gamma(d/2.))
    return y_0
    
def W_00naught(d):
    I = dblquad(lambda x,y: ((16.*(gamma(d))**2)/((gamma(d/2.))**4))*((math.cos(x))**(2*d))*((math.tan(x))**(d-1))*((math.cos(y))**(2*d+1))*(math.sin(y)), \
    0,math.pi/2, lambda x:0, lambda x:x)
    return I
    
def v_0(d):
    return 2*gamma(float(d))/((d+1.)*(gamma(d/2.))**2)
    
def c(i):
    return 2*math.sqrt(d-2)*math.sqrt(math.factorial(i + d -1)/math.factorial(i))/gamma(d/2.)

###############################################################################
###############################################################################

# Recursion relation for chi that takes rt.symmat(L) objects
def chi(x):
    for i in range(1,x.dim):
        for j in range(0,i+1):
            for k in range(0,j+1):
                for l in range(0,k+1):
                    x.T[i][j][k][l] = (((2*(w(i-1)+1))/(math.sqrt((i)*(i-1.+d))))*(1./(2+w(i-1) \
                    + w(j) + w(k) + w(l))))*(((d-1)/2.)*((w(j)**2)/(w(j)-1.) + (w(k)**2)/(w(k)-1) \
                    + (w(l)**2)/(w(l)-1) - ((w(i-1)**2)/(w(i-1)**2 -1))*(1+w(j)+w(k)+w(l)))*x.getel(i-1,j,k,l) \
                    + ((w(j)*math.sqrt(j*(j+d-1.)))/(w(j)-1))*x.getel(i-1,j-1,k,l) \
                    + ((w(k)*math.sqrt(k*(k+d-1.)))/(w(k)-1))*x.getel(i-1,j,k-1,l) \
                    + ((w(l)*math.sqrt(l*(l+d-1.)))/(w(l)-1))*x.getel(i-1,j,k,l-1) \
                    - (2+w(j)+w(k)+w(l)-w(i-1.))*((math.sqrt((i-1.)*(i+d-2)))/(2*(w(i-1)-1)))*x.getel(i-2,j,k,l))
        if i%50 == 0:
            print("Completed up to L = %d" % i)
    return x    

# Recursion relation for psi that takes rt.symmat(L) objects
def psi(y):
    for i in range(1,y.dim):
        for j in range(0,i+1):
            for k in range(0,j+1):
                for l in range(0,k+1):
                    y.T[i][j][k][l] = 2*(w(i-1)+1)*(-(d-1)*((w(i-1)**2 -2 -w(j)-w(k)-w(l))/(w(i-1)**2 -1.) - 6 + w(j)/(w(j)-1.) \
                    + w(k)/(w(k)-1) + w(l)/(w(l)-1))*y.getel(i-1,j,k,l)/2. + (w(i-1)-w(j)-w(k)-w(l)-2)*math.sqrt((i-1)*(i+d-2.))*y.getel(i-2,j,k,l)/(2*(w(i-1)-1.)) \
                    + w(j)*math.sqrt(j*(j+d-1.))*y.getel(i-1,j-1,k,l)/(w(j)-1) \
                    + w(k)*math.sqrt(k*(k+d-1.))*y.getel(i-1,j,k-1,l)/(w(k)-1) \
                    + w(l)*math.sqrt(l*(l+d-1.))*y.getel(i-1,j,k,l-1)/(w(l)-1))/((2.+w(i-1)+w(j)+w(k)+w(l))*math.sqrt(i*(i+d-1.)))
        if i%25 == 0:
            print("Completed up to L = %d" % i)
    return y
    
# Use chi to compute X
def makeX(x):
    for i in range(0,X.dim):
        for j in range(0,X.dim):
            for k in range(0,j+1):
                for l in range(0,k+1):
                    try:
                        X.T[i][j][k][l] = w(i)*(math.sqrt((i+1.)*(i+d))*x.getel(i+1,j,k,l)/(2*(w(i)+1)) \
                        - math.sqrt(i*(i+d-1.))*x.getel(i-1,j,k,l)/(2*(w(i)-1.)) \
                        - (d-1)*w(i)*x.getel(i,j,k,l)/(2*(w(i)**2 - 1.)))
                    except IndexError:
                        #print("Index error for X.T[%d][%d][%d][%d]" % (i,j,k,l))
                        pass
    return X
    
# Use phi to compute Y
def makeY(y):
    for i in range(0,Y.dim):
        for j in range(0,Y.dim):
            for k in range(0,Y.dim):
                for l in range(0,k+1):
                    try:
                        Y.T[i][j][k][l] = w(i)*w(k)*w(l)*(math.sqrt(j*(j+d-1.))*y.getel(i,j-1,k,l)/(2*(w(j)-1.)) \
                        - math.sqrt((j+1.)*(j+d))*y.getel(i,j+1,k,l)/(2*(w(j)+1.)) \
                        -(d-1)*w(j)*y.getel(i,j,k,l)/(2*(w(j)**2 -1.)))
                    except IndexError:
                        #print("Index error for Y.T[%d][%d][%d][%d]" % (i,j,k,l))
                        pass
    return Y

# Use X,Y to compute S
""" Verified against integrals and Recursion.mw""" 
def makeS(X,Y):
    for i in range(0,S.dim):
        for j in range(0,S.dim):
            for k in range(0,S.dim):
                for l in range(0,S.dim):
                    if i ==k or j==k:
                        pass
                        #print("S[%d][%d][%d][%d] does not exist due to restricted sum" % (i,j,k,l))
                    else:
                        S.T[i][j][k][l] = -(1/(w(i)+w(j)) + 1/(w(i)-w(k)) + 1/(w(j)-w(k)))*(w(i)*w(j)*w(k)*X.getel3(l,i,j,k) - w(l)*Y.getel2(i,l,j,k))/4. \
                        -(1/(w(i)+w(j)) + 1/(w(i)-w(k)) - 1/(w(j)-w(k)))*(w(j)*w(k)*w(l)*X.getel3(i,j,k,l) - w(i)*Y.getel2(j,i,k,l))/4. \
                        -(1/(w(i)+w(j)) - 1/(w(i)-w(k)) + 1/(w(j)-w(k)))*(w(i)*w(k)*w(l)*X.getel3(j,i,k,l) - w(j)*Y.getel2(i,j,k,l))/4. \
                        -(1/(w(i)+w(j)) - 1/(w(i)-w(k)) - 1/(w(j)-w(k)))*(w(i)*w(j)*w(l)*X.getel3(k,i,j,l) - w(k)*Y.getel2(i,k,j,l))/4.
    return S

# W_00 requires different recursion relations for k=l=0. Build these first.                    
def makeW_00_zeros():
    for i in range(0,W_00.dim):
        for j in range(0,W_00.dim):
            try:
                #print("W_00.T =", W_00.T, "\n")                
                #print("W_00.T[%d][%d][%d][%d] =" % (i,j,0,0), W_00.getelW(i,j,0,0), "\n")
                #print("W_00.T[%d][%d][%d][%d] =" % (j,i,0,0), W_00.getelW(j,i,0,0), "\n")
                #print("W_00.T[%d][%d][%d][%d] = %f" % (i-1,j,0,0,W_00.getelW(i-1,j,0,0)), "\n")
                #print("W_00.T[%d][%d][%d][%d] = %f" % (i,j-1,0,0,W_00.getelW(i,j-1,0,0)), "\n")
                #print("Calculating W_00.T[%d][%d][%d][%d]" % (i+1,j,0,0), "\n")
                W_00.T[i+1][j][0][0] = 2*(w(i)+1)*((d-1)*((w(i)**2 - w(j) - 4)/(w(i)**2 -1) + w(j)/(w(j)-1))*W_00.getelW(i,j,0,0)/2 \
                + math.sqrt(i*(i+d-1))*(w(i)-w(j)-4)*W_00.getelW(i-1,j,0,0)/(2*(w(i)-1)) \
                + math.sqrt(j*(j+d-1))*w(j)*W_00.getelW(i,j-1,0,0)/(w(j)-1) \
                - math.sqrt(d)*(X.getel3(0,i,j,1) - X.getel3(1,i,j,0))/(w(1)**2 - w(0)**2))/((w(i)+w(j)+4)*math.sqrt((i+1)*(i+d)))
                
                W_00.T[j+1][i][0][0] = W_00.T[i+1][j][0][0]
            except IndexError:
                print("Index error for W_00.T[%d][%d][0][0]" % (i+1,j), "\n")
                pass
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
                            #print("W_00.T[%d][%d][%d][%d] =" % (i,j,k,k), W_00.getelW(i,j,k,k), "\n")
                            #print("Calculating W_00.T[%d][%d][%d][%d]" % (i,j,k+1,k+1), "\n")
                            W_00.T[i][j][k+1][k+1] = W_00.getelW(i,j,k,k) \
                            + ((w(k)+1)*(d-1)/math.sqrt((k+1)*(k+d)))*(1/(w(k+1)**2 -1) - 1/(w(k)**2 -1))*(X.getel3(k+1,i,j,k) - X.getel3(k,i,j,k+1))/(w(k)**2 - w(k+1)**2) \
                            + ((w(k)+1)*(math.sqrt((k+2)*(k+d+1)))/((w(k+1)+1)*math.sqrt((k+1)*(k+d))))*(X.getel3(k+2,i,j,k)-X.getel3(k,i,j,k+2))/(w(k)**2 - w(k+2)**2) \
                            - ((w(k)+1)*math.sqrt(k*(k+d-1))/((w(k)-1)*math.sqrt((k+1)*(k+d))))*(X.getel3(k+1,i,j,k-1) - X.getel3(k-1,i,j,k+1))/(w(k-1)**2 - w(k+1)**2)
                            
                            W_00.T[j][i][k+1][k+1] = W_00.T[i][j][k+1][k+1]
                        except IndexError:
                            #print("W.T[%d][%d][%d][%d] does not exist" % (i,j,k+1,k+1))
                            pass
                    else:
                        try:
                            W_00.T[i][j][k][l] = (X.getel3(l,i,j,k) - X.getel3(k,i,j,l))/(w(k)**2 - w(l)**2)
                        except IndexError:
                            print("Index error for W_00.T[%d][%d][%d][%d]" % (i,j,k,l))
                            pass
    return W_00

def makeW_10():
    for i in range(0,W_10.dim):
        for j in range(0,W_10.dim):
            for k in range(0,W_10.dim):
                for l in range(0,k+1):
                    W_10.T[i][j][k][l] = (1/2)*(w(i)**2 + w(j)**2 -4)*W_00.getelW(i,j,k,l) - (d-1)*x.getel(i,j,k,l) \
                    - (X.getel3(i,j,k,l) + X.getel3(j,i,k,l)) - (1/2)*(X.getel3(k,i,j,l) + X.getel3(l,i,j,k))
    return W_10                    
    
def makeT():
    T = [None]*L
    for i in range(0,L):
        T[i] = (w(i)**2)*X.getel3(i,i,i,i)/2 + 3*Y.getel2(i,i,i,i)/2 + 2*(w(i)**4)*W_00.getelW(i,i,i,i) + 2*(w(i)**2)*W_10.getelW(i,i,i,i) \
        - (w(i)**2)*(A.getel2Dsym(i,i) + (w(i)**2)*V.getel2Dsym(i,i))
    return T
    
def makeR():
    for i in range(0,R.dim):
        for j in range(0,R.dim):
            if i == j:
                pass
                #print("R[%d][%d] does not exist due to restricted sum" % (i,j))
            else:
                R.B[i][j] = ((w(i)**2 + w(j)**2)/(w(j)**2 - w(i)**2))*((w(j)**2)*X.getel3(i,j,j,i) - (w(i)**2)*X.getel3(j,i,i,j))/2 \
                + 2*((w(j)**2)*Y.getel2(i,j,i,j) - (w(i)**2)*Y.getel2(j,i,j,i))/(w(j)**2 - w(i)**2) \
                + (Y.getel2(i,i,j,j) + Y.getel2(j,j,i,i))/2 \
                +(w(i)**2)*(w(j)**2)*(X.getel3(i,j,j,i) - X.getel3(j,i,j,i))/(w(j)**2 - w(i)**2) \
                + (w(i)**2)*(w(j)**2)*(W_00.getelW(j,j,i,i) + W_00.getelW(i,i,j,j)) \
                + (w(i)**2)*(W_10.getelW(j,j,i,i)) + (w(j)**2)*(W_10.getelW(i,i,j,j)) \
                - (w(j)**2)*(A.getel2Dsym(i,i) + (w(i)**2)*V.getel2Dsym(i,i))
    return R


""" Verified against integrals and PIvals.py"""    
def makeV():
    for i in range(1,V.dim):
        for j in range(0,i+1):
            V.B[i][j] = 2.*(w(i-1)+1)*((d-1)*((w(i-1)**2 - w(j)-4)/(w(i-1)**2 -1) + w(j)/(w(j)-1))*V.getel2Dsym(i-1,j)/2. \
            + (w(i-1) - w(j) -4)*math.sqrt((i-1)*(i+d-2.))*V.getel2Dsym(i-2,j)/(2.*(w(i-1)-1)) \
            + w(j)*math.sqrt(j*(j+d-1.))*V.getel2Dsym(i-1,j-1)/(w(j)-1.))/(math.sqrt(i*(i+d-1.))*(w(i-1)+w(j)+4.)) 
    return V

""" Verified against integrals and PIvals.py"""     
def makeA():
    for i in range(0,L):
            A.B[i][i] = (w(i)**2 + w(i)**2 -4)*V.getel2Dsym(i,i)/2 -c(i)*c(i)/2
    return A

# Create output files of entries for X,Y,R,T listed by index value with 14 significant figures    
def outputs(X,Y,R,T,S):
    
    print("Caculating X,Y,R,T from the recursion relations using L=%d and d=%d" % (L,d))
    
    with open("d3T.dat","w") as t:
        for i in range(0,len(T)):
            t.write("%d %.14e \n" %(i,T[i]))
        print("Wrote T to %s" % t.name)
        
    with open("d3R.dat","w") as r:
        for i in range(0,R.dim):
            for j in range(0,R.dim):
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
            for j in range(0,Y.dim):
                for k in range(0,j+1):
                    for l in range(0,k+1):
                        y.write("%d %d %d %d %.14e \n" % (i,j,k,l,Y.T[i][j][k][l]))
        print("Wrote Y to %s" % y.name)
        
    with open("d3S.dat","w") as s:
        for i in range(0,S.dim):
            for j in range(0,S.dim):
                for k in range(0,S.dim):
                    for l in range(0,S.dim):
                        try:
                            s.write("%d %d %d %d %.14e \n" % (i,j,k,l,S.T[i][j][k][l]))
                        except TypeError:
                            s.write("%d %d %d %d None \n" % (i,j,k,l))
        print("Wrote S to %s" % s.name)
    print("Done")
                    
######################################################################
######################################################################
     
"""
Maximum "level" to be calculated, L (non-inclusive), and number of dimensions, d
"""
L=7
d=3

"""
Chi and psi must be calculated to level L+3
"""

Chi = rt.symmat(L+3)
Chi.build()
Chi.T[0][0][0][0] = x_0(d)
chi(Chi)
#print("x.T =", x.T,"\n")


Psi = rt.symmat(L+3)
Psi.build()
Psi.T[0][0][0][0] = y_0(d)
psi(Psi)


#print("y.T =", y.T, "\n")

"""
Using chi and psi, X and Y are computed to level L+2
"""

X = rt.symmat(L+2)
X.build3()
makeX(Chi)
#print("X =", X.T, "\n")

Y = rt.symmat(L+2)
Y.build2()
makeY(Psi)
#print("Y =", Y.T,"\n")



"""
S is computed to level L+2 using X and Y. Note that values prohibited by the restricted
sums from equation 7 in ArXiv:1508.04943 will remain as "None" in S 
"""

S = rt.symmat(L+2)
S.buildnone()
makeS(X,Y)
"""
for i in range(0,S.dim):
        for j in range(0,S.dim):
            for k in range(0,S.dim):
                for l in range(0,S.dim):
                    if i ==k or j==k:
                        pass
                    else:
                        print("S[%d][%d][%d][%d] = %f" % (i,j,k,l,S.T[i][j][k][l]), "\n")
print("S =", S.T,"\n")
"""

"""
Both R and T require calculating W_00 and W_10 first; W_00 can only be calculated to level L
"""

# W_00 is computed to level L
W_00 = rt.symmat(L)
"""W_00.build2()
W_00.T[0][0][0][0] = W_00naught(d)[0]
makeW_00_zeros()
#print("W_00_zeros =", W_00.T, "\n") 
makeW_00()
#print("W_00 =", W_00.T, "\n")

# W_10 is computed to level L
W_10 = rt.symmat(L)
W_10.build2()
makeW_10()
#print("W_10[%d][%d][%d][%d] =" % (0,0,0,0), W_10.getelW(0,0,0,0), "\n")
#print("W_10[%d][%d][%d][%d] =" % (1,1,0,0), W_10.getelW(1,1,0,0), "\n")
#print("W_10[%d][%d][%d][%d] =" % (1,1,1,1), W_10.getelW(1,1,1,1), "\n")
#print("W_10[%d][%d][%d][%d] =" % (2,2,0,0), W_10.getelW(2,2,0,0), "\n")
#print("W_10[%d][%d][%d][%d] =" % (2,2,1,1), W_10.getelW(2,2,1,1), "\n")
#print("W_10[%d][%d][%d][%d] =" % (2,2,2,2), W_10.getelW(2,2,2,2), "\n")

"""
"""
Additional coefficients in the interior time gauge, V and A are calculated up to level L
"""
"""
V = rt.symmat(L)
V.build2Dsym()
V.B[0][0] = v_0(d)
makeV()
#print("V =",V.B,"\n")
A = rt.symmat(L)
A.build2Dsym()
makeA()
#print("A =",A.B,"\n")

"""
#Now use the results of W_00 and W_10 to calculate R and T up to level L 

# T is computed to level L
#T = makeT()
#print("T =", T, "\n")
# R is computed to level L
#R = rt.symmat(L)
#R.build2D()
#makeR()
#print("R =", R.B, "\n") 


#Finally, output the results into individual files

#outputs(X,Y,R,T,S)

with open("d3S_L10.dat","w") as s:
        for i in range(0,S.dim):
            for j in range(0,S.dim):
                for k in range(0,S.dim):
                    for l in range(0,S.dim):
                        try:
                            s.write("%d %d %d %d %.14e \n" % (i,j,k,l,S.T[i][j][k][l]))
                        except TypeError:
                            s.write("%d %d %d %d None \n" % (i,j,k,l))
        print("Wrote S to %s" % s.name)


with open("d3X_L10.dat","w") as x:
       for i in range(0,X.dim):
           for j in range(0,i+1):
               for k in range(0,j+1):
                   for l in range(0,k+1):
                       x.write("%d %d %d %d %.14e \n" % (i,j,k,l,X.T[i][j][k][l]))
       print("Wrote X to %s" % x.name)    
       
with open("d3Y_L10.dat","w") as y:
       for i in range(0,Y.dim):
           for j in range(0,Y.dim):
               for k in range(0,j+1):
                   for l in range(0,k+1):
                       y.write("%d %d %d %d %.14e \n" % (i,j,k,l,Y.T[i][j][k][l]))
       print("Wrote Y to %s" % y.name)           
       
       
"""
with open("/Users/bradc/Documents/University_of_Manitoba/Research_2015/TTF/TtfEvolution/Psi_50.dat","w") as y:
       for i in range(0,Psi.dim):
           for j in range(0,i+1):
               for k in range(0,j+1):
                   for l in range(0,k+1):
                       y.write("%d %d %d %d %.14e \n" % (i,j,k,l,Psi.T[i][j][k][l]))
       print("Wrote Psi to %s" % y.name)    
       
       
with open("/Users/bradc/Documents/University_of_Manitoba/Research_2015/TTF/TtfEvolution/Chi_50.dat","w") as x:
       for i in range(0,Chi.dim):
           for j in range(0,i+1):
               for k in range(0,j+1):
                   for l in range(0,k+1):
                       x.write("%d %d %d %d %.14e \n" % (i,j,k,l,Chi.T[i][j][k][l]))
       print("Wrote Chi to %s" % x.name)           
"""
       
       
       
       
       
       
       
       
       
       
       
       
       
       