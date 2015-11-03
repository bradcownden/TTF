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
 
# Build a list of empty entries corresponding to a totally
# symmetric 4D matrix
        
    def build(self):
        self.T = [None]*(self.dim)
        for i in range(0,self.dim):
            self.T[i] = [None]*(i+1)
            for j in range(0,i+1):
                self.T[i][j] = [None]*(j+1)
                for k in range(0,j+1):
                    self.T[i][j][k] = [None]*(k+1)
        return self.T

# Build a list of empty entries correponding to a 4D matrix
# that is symmetric in the last three indices
        
    def build3(self):
        self.T = [None]*(self.dim)
        for i in range(0,self.dim):
            self.T[i] = [None]*(self.dim)
            for j in range(0,self.dim):
                self.T[i][j] = [None]*(j+1)
                for k in range(0,j+1):
                    self.T[i][j][k] = [None]*(k+1)

# Build a list of empty entries corresponding to a 4D matrix
# that is symmetric in the last two indices
                    
    def build2(self):
        self.T = [None]*(self.dim)
        for i in range(0,self.dim):
            self.T[i] = [None]*(self.dim)
            for j in range(0,self.dim):
                self.T[i][j] = [None]*(self.dim)
                for k in range(0,self.dim):
                    self.T[i][j][k] = [None]*(k+1)

# Buid a list of empty entries corresponding to a 4D matrix
# that has no symmetries

    def buildnone(self):
        self.T = [None]*(self.dim)
        for i in range(0,self.dim):
            self.T[i] = [None]*(self.dim)
            for j in range(0,self.dim):
                self.T[i][j] = [None]*(self.dim)
                for k in range(0,self.dim):
                    self.T[i][j][k] = [None]*(self.dim)
                    

# Build an LxL list   
     
    def build2D(self):
        self.B = [None]*(self.dim)
        for i in range(0,self.dim):
            self.B[i] = [None]*(self.dim)
        return self.B
        
# Build a completely symmetric LxL list
    
    def build2Dsym(self):
        self.B = [None]*(self.dim)
        for i in range(0,self.dim):
            self.B[i] = [None]*(i+1)
        return self.B

# Return the element of self.T when totally symmetric  
      
    def getel(self,a,b,c,d):
        temp = sorted([a,b,c,d], reverse=True)
        if a<0 or b<0 or c<0 or d<0:
            return 0
        else:
            return self.T[temp[0]][temp[1]][temp[2]][temp[3]] 

# Return the element of self.T when symmetric in the last three
# indices       
     
    def getel3(self,a,b,c,d):
        temp = sorted([b,c,d], reverse=True)
        if a<0 or b<0 or c<0 or d<0:
            return 0
        else:
            return self.T[a][temp[0]][temp[1]][temp[2]]

# Return the element of self.T when symmetric in the last two
# indices    
        
    def getel2(self,a,b,c,d):
        temp = sorted([c,d], reverse=True)
        if a<0 or b<0 or c<0 or d<0:
            return 0
        else:
            return self.T[a][b][temp[0]][temp[1]]
            
    def getel2Dsym(self,a,b):
        temp = sorted([a,b], reverse=True)        
        if a<0 or b<0:
            return 0
        else:
            return self.B[temp[0]][temp[1]]
            
############################################################################        
############################################################################        
            