
# coding: utf-8

# In[81]:


from __future__ import division
import numpy as np
import scipy
from scipy import linalg as sciLA
from scipy import integrate
from numpy import linalg as LA
from scipy.sparse import spdiags
from scipy.sparse import diags
import scipy.io as sio

import matplotlib.pyplot as plt


# #### Solving with Guassian or LU Decomp 

# In[82]:


A = np.matrix([[1,1,1],[1,2,4],[1,3,9]])
b = np.matrix([[1],[-1],[1]])

#simple way to solve, uses Guassian Elimination, similiar to a \ in matlab
x = LA.solve(A,b)
#print x

#LU decomposition for solving
[P, L, U] = sciLA.lu(A)
y = LA.solve(L, LA.inv(P) * b)
x = LA.solve(U,y)
#print x


# #### Import .mat  for HW

# In[83]:


mat_contents = sio.loadmat('Fmat.mat')
Fmat = mat_contents['Fmat']
Fmat.shape


# In[84]:


mat_contents = sio.loadmat('permvec.mat')
permvec = mat_contents['permvec']
permvec.shape


# #### Iterative Methods

# ##### Solve with Jacobi Method

# In[271]:


#given
A = np.matrix([[4,-1,1],
               [4,-8,1],
               [-2,1,5]])

b = np.matrix([[7],[-21],[15]])

#first we would want to check for strictly diagonal dominant

#create diagonal matrix
diag = np.diag(A)
D = np.diag(diag)
# create R matrix
R = A - D
R_sums = R.sum(axis=1)

#loop through and check for SDD
for i in range(0, len(diag)):
    if abs(R_sums[i]) > abs(diag[i]):
        print 'fails at index: ' + str(i)
    i = i + 1
    
    
#create other matrices needed to solve

#create strict upper matrix
Us = np.asmatrix(np.triu(A, k=1))
#create upper matrix
U = np.asmatrix(np.triu(A, k=0))
#create lower matrix
L = np.asmatrix(np.tril(A, k=0))

x0 = [1, 2, 2]
x = np.column_stack((x0,))

#set tolerance
tol = 10**-4

for i in range(0, 10):
    x1 = LA.inv(D) * (-R * x + b)
    
    rx = -R*x
    rxb = (-R * x + b)
    
    if np.allclose(x, x1, rtol=1e-4):
        print 'it took ' + str(i+1) + ' iterations to find a solution'
        break;
    
    x = x1


# ##### Solve with Gauss Seidel Method

# In[273]:


#given
A = np.matrix([[4,-1,1],
               [4,-8,1],
               [-2,1,5]])

b = np.matrix([[7],[-21],[15]])

#first we would want to check for strictly diagonal dominant

#create diagonal matrix
diag = np.diag(A)
D = np.diag(diag)
# create R matrix
R = A - D
R_sums = R.sum(axis=1)

#loop through and check for SDD
for i in range(0, len(diag)):
    if abs(R_sums[i]) > abs(diag[i]):
        print 'fails at index: ' + str(i)
    i = i + 1
    
    
#create other matrices needed to solve
    
#LU decomposition for solving
[P, L, U] = sciLA.lu(A)

#create strict upper matrix
Us = np.asmatrix(np.triu(U, k=1))
#create upper matrix
U = np.asmatrix(np.triu(U, k=0))
#create lower matrix
L = np.asmatrix(np.tril(L, k=0))

x0 = [1, 2, 2]
x = np.column_stack((x0,))

#set tolerance
tol = 10**-4

for i in range(0, 100):
    
    L_inv = LA.inv(L)
    Usb = (-Us + b)
    
    x1 = (LA.inv(L) * (-Us + b) * x)
    
    if np.allclose(x, x1, tol):
        print 'it took ' + str(i+1) + ' iterations to find a solution'
        break;
    
    x = x1


# In[274]:


x1 = LA.inv(L) * ((-Us + b) * x)


# In[270]:

ITERATION_LIMIT = 10

#given
A = np.matrix([[4,-1,1],
               [4,-8,1],
               [-2,1,5]])

b = np.matrix([[7],[-21],[15]])

x = np.zeros_like(b)
for it_count in range(1, ITERATION_LIMIT):
    x_new = np.zeros_like(x)
    print("Iteration {0}: {1}".format(it_count, x))
    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x_new[:i])
        s2 = np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
    if np.allclose(x, x_new, rtol=1e-4):
        break
    x = x_new

print("Solution: {0}".format(x))
error = np.dot(A, x) - b
print("Error: {0}".format(error))

