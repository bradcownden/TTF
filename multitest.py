# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:37:36 2016

@author: bradc
"""

import multiprocessing, time, random, math
from numba import jit
import numpy as np
from scipy.optimize import newton_krylov



S = np.genfromtxt("d3S_L10.dat",dtype=np.float)
R = np.genfromtxt("Mathematica_R.dat",dtype=np.float)
T = np.genfromtxt("Mathematica_T.dat",dtype=np.float)


@jit()
def w(n,d):
    if n<0:
        return float(d)
    else:
        return float(d)+2.*n


@jit(nopython=True,nogil=True)
def Sval(i,j,k,l):
    if i<0 or j<0 or k<0 or l<0:
        return 0
    else:
        for row in range(S.shape[0]):
            if S[row][0]==i and S[row][1]==j and S[row][2]==k and S[row][3]==l:
                return S[row][4]

@jit(nopython=True,nogil=True)     
def Rval(i,j):
    for row in range(R.shape[0]):
        if R[row][0]==i and R[row][1]==j:
            return R[row][2]    
    return 0

@jit(nopython=True,nogil=True)    
def Tval(i):
    for row in range(T.shape[0]):
        if T[row][0]==i:
            return T[row][1]


a0 = float(1.0)
a1 = float(0.35)
N = T.shape[0]
d = 3.


# Initial values for each alpha_j based on (B1) of arXiv:1507.08261
@jit(nogil=True)
def ainits(a1):
    mu = math.log(3./(5.*a1))
    ainits = [float(1.0e2),float(1.0e2)]
    for i in range(2,N):
        ainits.append(3.*math.exp(-mu*float(i))/(2.*float(i)+3.))
    return ainits
    
@jit(nopython=True,nogil=True)
def alpha(i,x):
    if i==0:
        return a0
    if i==1:
        return a1
    if i>1:
        return x[i]


# Use the QP equation (14) from arXiv:1507.08261
@jit(nogil=True)
def system(x):
    F = np.zeros_like(x,dtype=np.float)
    for i in range(N):
        s = 0; r = 0
        for j in range(N):
            if i!=j:
                for k in range(N):
                    if j+k-i<N and i<j+k and i!=k:
                        s = s +2.*Sval(j,k,j+k-i,i)*alpha(j,x)*alpha(k,x)*alpha(j+k-i,x) 
                r = r + 2.*Rval(i,j)*alpha(i,x)*alpha(j,x)**2
        F[i] = 2.*Tval(i)*alpha(i,x)**3 + w(i,d)*(x[0]+float(i)*(x[1]-x[0]))*alpha(i,x) + r + s
    return F
    
# Compute the energy per mode using (5) from arXiv:1507.08261
@jit(nogil=True)
def energy(x):
    E = np.zeros_like(x,dtype=np.float)    
    E[0]= 4.*w(0,d)**2*a0**2
    E[1] = 4.*w(1,d)**2*a1**2
    for i in range(2,len(x)):
        E[i] = 4.*w(i,d)**2*x[i]**2
    return E

@jit(nogil=True)
def solves():
    t0 = time.process_time()
    sol = newton_krylov(system,ainits,verbose=True,f_tol=1.0e-20,f_rtol=1.0e-10)
    print(sol)
    print(energy(sol))
    t1 = time.process_time()
    print("Calculation time =",t1-t0,"seconds")
    print('\a')
    return sol
    
sol = solves()



# Functions used in tasks
def f(a,b):
    time.sleep(0.5*random.random())
    return a+b

def g(a,b):
    time.sleep(0.5*random.random())
    return a*b

def h(x):
    return x**2
    
def noop(x):
    print("You've been 'nooped'!")
  


# Function used to calculate result
def calculate(func,args):
    result = func(*args)
    return "%s says that %s%s = %s" % (multiprocessing.current_process().name,\
    func.__name__, args, result)
    
def calculatestar(args):
    return calculate(*args)
    

# Actual code
def test():
    numpros = 4
    print("Creating pool with %d processes \n" % numpros)
    
    with multiprocessing.Pool(numpros) as pool:
        tasks = [(f,(i,2)) for i in range(10)]+[(g,(i,5)) for i in range(10)]
        
        results = [pool.apply_async(calculate,t) for t in tasks]
        imap_it = pool.imap(calculatestar,tasks)
        imap_unordered_it = pool.imap_unordered(calculatestar, tasks)
        
        print("Ordered results using pool.apply_async:")
        for r in results:
            print("\t",r.get())
        print()
        
        print("Ordered results using pool.imap:")
        for x in imap_it:
            print("\t",x)
        print()
        
        print("Unordered results from pool.imap_unordered:")
        for x in imap_unordered_it:
            print("\t",x)
        print()
        
        print("Ordered results using pool.map -- will block untill complete:")
        for x in pool.map(calculatestar,tasks):
            print("\t",x)
        print()
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    test()