# -*- coding: utf-8 -*-
"""
Created on Mon Nov 05 14:28:58 2018

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
vbc1 = (N-1)*M #top wrap
dbc1 = np.ones(dims)

vbc2 = -(N-1)*M #bottom wrap
dbc2 = np.ones(dims)

#central conditions
v2 = -N #value below
d2 = np.ones(dims)

v3 = N #value above
d3 = np.ones(dims)

data = np.array([d2,d3,dbc1,dbc2])
diag = np.array([v2,v3,vbc1,vbc2])

C = spdiags(data, diag, dims, dims).toarray()