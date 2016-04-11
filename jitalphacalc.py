# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:31:20 2016

@author: bradc
"""
"""
Numerically solve the ode for the complex coefficients of the scalar solution in the 
quasistable regime using the recursion coefficients. Test up to L=10.
"""

import numpy as np
#from scipy.integrate import quad, dblquad
#from scipy.special import gamma
#from scipy.special import eval_jacobi
import math
#from sympy import lambdify, jacobi
#from sympy.functions.elementary.trigonometric import cos as symcos
#from sympy.core import diff
#from sympy.abc import t
from scipy.optimize import newton_krylov, diagbroyden, fsolve, root
import time
#from array import array
from numba import jit




###################################################################
###################################################################

"""
Read in values for X,Y,S from recursion relations.
"""

#S = np.genfromtxt("d3S_L10.dat",dtype=np.float)
#X = np.genfromtxt("d3X_L10.dat")
#Y = np.genfromtxt("d3Y_L10.dat")

"""
Read in T,R from Mathematica output
"""

#R = np.genfromtxt("Mathematica_R.dat",dtype=np.float)
#T = np.genfromtxt("Mathematica_T.dat",dtype=np.float)


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

@jit(nogil=True)
def Tsort(Tbin):
    T = np.sort(Tbin)
    Tten = np.zeros((1,2),dtype=np.float)
    for i in range(Tbin.shape[0]):
        Tten = np.vstack((Tten,np.array([T[i][0],T[i][1]],dtype=np.float)))
    return Tten[1:]
    

@jit(nogil=True)
def Rsort(Rbin):
    R = np.sort(Rbin)
    Rten = np.zeros((1,3),dtype=np.float)
    for i in range(Rbin.shape[0]):
        Rten = np.vstack((Rten,np.array([R[i][0],R[i][1],R[i][2]],dtype=np.float)))
    return Rten[1:]


@jit(nogil=True)
def Ssort(Sbin):
    S = np.sort(Sbin)
    Sten = np.zeros((1,5),dtype=np.float)
    for i in range(Sbin.shape[0]):
        Sten = np.vstack((Sten,np.array([S[i][0],S[i][1],S[i][2],S[i][3],S[i][4]],dtype=np.float)))
    return Sten[1:]


T = Tsort(Tbin)
R = Rsort(Rbin)
S = Ssort(Sbin)


print("Data read-in complete \n")

###################################################################
###################################################################
  
"""
Basis functions to be used in computing other tensors
"""
@jit()
def w(n,d):
    if n<0:
        return d
    else:
        return d+np.longdouble(2.*n)

@jit(nogil=True,nopython=True)
def Sval(i,j,k,l):
    for row in range(S.shape[0]):
        if S[row][0]==i and S[row][1]==j and S[row][2]==k and S[row][3]==l:
            return S[row][4]

@jit(nogil=True,nopython=True)
def Rval(i,j):
    for row in range(R.shape[0]):
        if R[row][0]==i and R[row][1]==j:
            return R[row][2]


@jit(nogil=True,nopython=True)
def Tval(i):
    for row in range(T.shape[0]):
        if T[row][0]==i:
            return T[row][1]


"""
These tensors will be calculated by integrals elsewhere before being used; for the test case,
generating the values as needed is sufficient.
""" 
""" 
def W00(i,j,k,l,d):
    return dblquad(lambda x,y: (ks(i,d)*((math.cos(x))**d)*eval_jacobi(i,0.5*d -1,0.5*d,math.cos(2*x))) \
    * (ks(j,d)*((math.cos(x))**d)*eval_jacobi(j,0.5*d -1,0.5*d,math.cos(2*x))) * math.sin(x)*math.cos(x) \
    * (ks(l,d)*((math.cos(y))**d)*eval_jacobi(l,0.5*d -1,0.5*d,math.cos(2*y))) \
    * (ks(k,d)*((math.cos(y))**d)*eval_jacobi(k,0.5*d -1,0.5*d,math.cos(2*y))) * (math.tan(y))**(d-1.), \
    0,math.pi/2, lambda x:0, lambda x:x)[0]
    
def W10(i,j,k,l,d):
    return dblquad(lambda x,y: (ep(i,d)(x))*(ep(j,d)(x)) * math.sin(x)*math.cos(x) \
    * (ks(l,d)*((math.cos(y))**d)*eval_jacobi(l,0.5*d -1,0.5*d,math.cos(2*y))) \
    * (ks(k,d)*((math.cos(y))**d)*eval_jacobi(k,0.5*d -1,0.5*d,math.cos(2*y))) * (math.tan(y))**(d-1.), \
    0,math.pi/2, lambda x:0, lambda x:x)[0]
    
def A(i,j,d):
    return quad(lambda x: (ks(i,d)*((math.cos(x))**d)*eval_jacobi(i,0.5*d -1,0.5*d,math.cos(2*x))) \
    * (ks(j,d)*((math.cos(x))**d)*eval_jacobi(j,0.5*d -1,0.5*d,math.cos(2*x))) * math.sin(x)*math.cos(x), \
    0, math.pi/2.)[0]
    
def V(i,j,d):
    return quad(lambda x: (ep(i,d)(x))*(ep(j,d)(x))*math.sin(x)*math.cos(x),0,math.pi/2.)[0]
    
"""
###################################################################
###################################################################

"""
Inputs for the QP mode solver
"""

# Initial values for alpha_0 and alpha_1; maximum number N = j_max; dimension d
a0 = np.longdouble(1.0)
a1 = np.longdouble(0.35)
N = T.shape[0]
d = np.longdouble(3.)


# Initial values for each alpha_j based on (B1) of arXiv:1507.08261
@jit(nogil=True)
def inits(a0,a1):
    mu = math.log(3./(5.*a1))
    inits = [a0,a1]
    for i in range(2,N):
        inits.append(np.float(3.*math.exp(-mu*float(i))/(2.*float(i)+3.)))
    return inits

ainits = inits(a0,a1)
#print(ainits)

# Initial values for beta0 and beta1 based on alpha_j seeds
@jit(nogil=True)
def makebees(seeds):
    #print("Constructing beta")
    beta = [np.float(0.)]
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
        beta.append(np.float((-2.*Tval(i)*(seeds[i])**3 - r - s)/(w(i,d)*seeds[i])))
    beta = beta[1:]
    return beta
        
b = makebees(ainits)
#print("beta =",b)


# Seed for the optimizer needs b0,b1 to be the first entries since a0, a1 are fixed  
seeds = np.concatenate((b,ainits[2:]))
print("seeds =",seeds,"\n")  
  
###################################################################
###################################################################
  
  
  
"""    
Functions for solving for QP coefficients
"""

# Function that returns the input values a0, a1 when i=0,1 and returns the variable
# being optimized when i>1
@jit(nogil=True)
def alpha(i,x):
    if i==0:
        return a0
    if i==1:
        return a1
    if i>1:
        return np.longdouble(x[i])



# Use the QP equation (14) from arXiv:1507.08261
@jit(nogil=True)
def system(x):
    F = [np.longdouble(0.)]
    for i in range(N):
        r = np.longdouble(0.) ; s = np.longdouble(0.)
        for j in range(N):
            if i!=j:
                r = r + np.longdouble(2.*Rval(i,j)*alpha(i,x)*alpha(j,x)**2)
                for k in range(N):
                    if k+j-i<N and k!=i and j+k>=i:
                        s = s + np.longdouble(2.*Sval(j,k,j+k-i,i)*(alpha(j,x))*(alpha(k,x))*(alpha(j+k-i,x)))
        #print("r =",r)
        #print("s =",s)
        F.append(np.longdouble(2.*Tval(i)*(alpha(i,x))**3 + w(i,d)*(x[0]+float(i)*(x[1]-x[0]))*alpha(i,x) + r + s))
    return F[1:]
 

# Compute the energy per mode using (5) from arXiv:1507.08261
@jit(nogil=True)
def energy(x):
    E = np.zeros_like(x,dtype=np.longdouble)    
    E[0]= 4.*w(0,d)**2*a0**2
    E[1] = 4.*w(1,d)**2*a1**2
    for i in range(2,len(x)):
        E[i] = 4.*w(i,d)**2*x[i]**2
    return E


###################################################################
###################################################################


"""
Solve for QP coefficients
"""

@jit(nogil=True)
def Ksolves():
    t0 = time.process_time()
    print("Calculating alphas with newton_krylov method")
    sol = newton_krylov(system,seeds,verbose=True,f_rtol=1e-10)
    #print("sol")
    print("Krylov method \n",sol)
    #print(energy(sol))
    print("Krylov calculation time =",time.process_time()-t0,"seconds")
    print('\a')
    return sol
    
#Ksol = Ksolves()

@jit(nogil=True)
def Bsolves():
    t0 = time.process_time()
    print("Calculating alphas with anderson method")
    sol = diagbroyden(system,seeds,f_rtol=1e-8)
    print("Anderson method \n",sol)
    print("Anderson calculation time =", time.process_time()-t0,"seconds")
    print('\a')
    return sol
 
#Bsol = Bsolves()
   
@jit(nogil=True)
def Fsolves():
    t0 = time.process_time()
    print("Calculating alphas with fsolve method")
    sol = fsolve(system,seeds,xtol=1e-15)
    print("fsolve method \n",sol)
    print("fsolve calculation time =", time.process_time()-t0,"seconds")
    print('\a')
    return sol

Fsol = Fsolves()    

@jit(nogil=True)
def Rsolves():
    t0 = time.process_time()
    print("Calculating alphas with root/krylov method")
    sol = root(system,seeds,method='krylov',tol=1e-15)
    print("root with krylov method \n",sol)
    print("root method calculation time =", time.process_time()-t0,"seconds")
    print('\a')
    return sol
    
Rsol = Rsolves()




with open("./data/AdS4QP_j10_a035_prime.dat","w") as s:
    s.write("%d %.14e \n" % (0,a0))
    s.write("%d %.14e \n" % (1,a1))
    for i in range(2,len(Fsol)):
        s.write("%d %.14e \n" % (i,Fsol[i]))
    s.write("beta0 %.14e \n" % Fsol[0])
    s.write("beta1 %.14e \n" % Fsol[1])
    print("Wrote QP modes to %s" % s.name)
                        
with open("./data/AdS4QP_j10_a035E_prime.dat","w") as f:
    for i in range(len(energy(Fsol))):
        f.write("%d %.14e \n" % (i,energy(Fsol)[i]))
    print("Wrote QP mode energies to %s" % f.name)




###################################################################
###################################################################


"""
Test case of N=3
"""

def f(x):
    f = np.zeros(3)    
    
    f[0] = 2.*Tval(0) + 2.*(Rval(0,1)*(0.2**2) + Rval(0,2)*x[2]**2) + w(0,d)*x[0] \
    + 2.*(Sval(0,1,1,0)*(0.2**2) + Sval(0,2,2,0)*x[2]**2 + Sval(1,0,1,0)*(0.2**2) \
    + Sval(1,1,2,0)*x[2]*(0.2**2) + Sval(2,0,2,0)*x[2]**2)
    
    f[1] = 2.*Tval(1)*(0.2**3) + 2.*(Rval(1,0)*0.2 + Rval(1,2)*0.2*x[2]**2) \
    + w(1,d)*x[1]*0.2 + 2.*(Sval(0,1,0,1)*0.2 + Sval(0,2,1,1)*0.2*x[2] + Sval(1,0,0,1)*0.2 \
    + Sval(1,1,1,1)*(0.2**3) + Sval(1,2,2,1)*(0.2)*x[2]**2 + Sval(2,0,1,1)*0.2*x[2] \
    + Sval(2,1,2,1)*0.2*x[2]**2)
    
    f[2] = 2.*Tval(2)*x[2]**3 + 2.*(Rval(2,0) + Rval(2,1)*0.2**2)*x[2] \
    + w(2,d)*(x[0] + 2.*(x[1]-x[0]))*x[2] + 2.*(Sval(0,2,0,2)*x[2] + Sval(1,1,0,2)*(0.2**2) \
    + Sval(1,2,1,2)*(0.2**2)*x[2] + Sval(2,0,0,2)*x[2] + Sval(2,1,1,2)*(0.2**2)*x[2] \
    + Sval(2,2,2,2)*(x[2]**3))
    
    return f
    
#x0 = np.ones(3)
#sol = newton_krylov(f,x0)
#print("Test case with explicit formulas [beta_0, beta_1, alpha_2] = [%.10e, %.10e, %.10e]" % (sol[0],sol[1],sol[2]))

###################################################################
###################################################################










