"""
Defines class symmat(dim). Use the symmat.build() function to create a list of 
symmetric entries in the self.T attribute 
To return any element of self.T, use symmat.getel(a,b,c,d)
Public functions are vals(L), dim(L), upto(L)
"""

import numpy as np

# The unique, symmetric combinations of four indices that add to L and are
# ordered with the largest value on the left
def vals(L):
    x=[]
    for i in range(L,-1,-1):
        for j in range(L-i,-1,-1):
            if i+j<= L and i>=j:
                for k in range(L-i-j,-1,-1):
                    if i+j+k <= L and j>=k:
                        for l in range(L-i-j-k,-1,-1):
                            if i+j+k+l == L and k >=l:
                                x.append([i,j,k,l])
    return x
    
# Gives the number of unique, symmetric combinations of indices for some L
def dim(L):
    x = np.array(vals(L))
    return x.shape[0]
        
def upto(L):
    x = [None]*(L+1)
    for i in range(0,L+1):
        x[i] = vals(i)
    return x


# Class of symmetric matrix objects
class symmat(list):
    """Use symmat.build() to construct a list of entries in a symmetric matrix
    Use symmat.getel(a,b,c,d) to retrieve the unique entry for any a,b,c,d
    N.B. if any of a,b,c,d are <0, symmat.getel returns 0""" 
    
    def __init__(self,dim):
        self.dim = dim
 
# Build a symmetric list of empty entries that can be referenced by other
# functions    
        
    def build(self):
        self.T = [None]*(self.dim)
        for i in range(0,self.dim):
            self.T[i] = [None]*(i+1)
            for j in range(0,i+1):
                self.T[i][j] = [None]*(j+1)
                for k in range(0,j+1):
                    self.T[i][j][k] = [None]*(k+1)
        return self.T
        
    def buildX(self):
        self.T = [None]*(self.dim)
        for i in range(0,self.dim):
            self.T[i] = [None]*(self.dim)
            for j in range(0,self.dim):
                self.T[i][j] = [None]*(j+1)
                for k in range(0,j+1):
                    self.T[i][j][k] = [None]*(k+1)
                    
    def buildY(self):
        self.T = [None]*(self.dim)
        print(self.T)
        for i in range(0,self.dim):
            self.T[i] = [None]*(self.dim)
            print(self.T)
            for j in range(0,self.dim):
                self.T[i][j] = [None]*(self.dim)
                print(self.T)
                for k in range(0,self.dim):
                    print(i,j,k,k+1)
                    self.T[i][j][k] = [None]*(k+1)
        
    def build2(self):
        self.B = [None]*(self.dim)
        for i in range(0,self.dim):
            self.B[i] = [None]*(i+1)
        return self.B

# Return the symmetric element of self.T[a][b][c][d]        
    def getel(self,a,b,c,d):
        temp = sorted([a,b,c,d], reverse=True)
        if a<0 or b<0 or c<0 or d<0:
            return 0
        else:
            return self.T[temp[0]][temp[1]][temp[2]][temp[3]] 
            
    def Xgetel(self,a,b,c,d):
        temp = sorted([b,c,d], reverse=True)
        if a<0 or b<0 or c<0 or d<0:
            return 0
        else:
            return self.T[a][temp[0]][temp[1]][temp[2]]
            
    def getel2(self,a,b):
        temp = sorted([a,b], reverse=True)
        return self.B[temp[0]][temp[1]]
            
############################################################################        
############################################################################        
            