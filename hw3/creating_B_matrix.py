# -*- coding: utf-8 -*-
"""
Created on Mon Nov 05 14:25:58 2018

@author: sspielman
"""

from __future__ import division
import numpy as np
from scipy.sparse import spdiags

#starting matrix to stack
N=8 # number of columns
M=8 # number of rows
dims = M*N

'''
NOTE: the way the scipy spdiags command works the diagonals line up 
with first column. So if the diag has values being replaced, it will
only need to be shifted if its position value is positive. The amount
of the shift is equal to the position value.
'''

#boundary conditions
vbc3 = N-1
dbc3 = np.ones(dims) 
for i in range(1,len(dbc3)): #loop through and zero out values
    if (i+vbc3+2) % (N) != 0: #I am not really sure what this 2 is doing. 
        dbc3[i] = 0
        
dbc4 = np.ones(dims)
vbc4 = -N+1
for i in range(1,len(dbc4)):
    if i % (N) != 0:
        dbc4[i] = 0

#central conditions

v4 = -1 #value left
d4 = np.ones(dims)
for i in range(1,len(d4)): #loop through and zero BC values
    if (i+abs(v4))%N == 0: #I don't fully understand how this shift is working
        d4[i] = 0

v5 = 1 #value right
d5 = np.ones(dims) #loop through and zero BC values
for i in range(1,len(d5)):
    if (i)%N == 0:
        d5[i] = 0

data = np.array([d4,d5,dbc3,dbc4])
diag = np.array([v4,v5,vbc3,vbc4])

B = spdiags(data, diag, dims, dims).toarray()