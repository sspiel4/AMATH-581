# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:06:15 2018

@author: sspie

hw3_main
"""
from __future__ import division
import numpy as np
from scipy.sparse import spdiags
import scipy.io as sio
from unscramble import unscramble


############### PROBLEM #1 ##########################

#starting matrix to stack
N=8 # number of columns
M=8 # number of rows
dims = M*N

#start with array of zeros
A = np.zeros((dims,dims))


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
v1 = 0 #center value
d1 = np.ones(dims)*-4

v2 = -N #value below
d2 = np.ones(dims)

v3 = N #value above
d3 = np.ones(dims)

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

#create A matrix
data = np.array([d1,d2,d3,d4,d5,dbc1,dbc2,dbc3,dbc4])
diag = np.array([v1,v2,v3,v4,v5,vbc1,vbc2,vbc3,vbc4])

A = spdiags(data, diag, dims, dims).toarray()

#create B matrix
data = np.array([d4,d5,dbc3,dbc4])
diag = np.array([v4,v5,vbc3,vbc4])

B = spdiags(data, diag, dims, dims).toarray()

#create C matrix
data = np.array([d2,d3,dbc1,dbc2])
diag = np.array([v2,v3,vbc1,vbc2])

C = spdiags(data, diag, dims, dims).toarray()

#save outputs
np.savetxt('A1.dat', A)
np.savetxt('A2.dat', B)
np.savetxt('A3.dat', C)



############### PROBLEM #2 ##########################

#load key
mat_contents = sio.loadmat('permvec.mat')
permvec = mat_contents['permvec']
[Hp, Wp] = permvec.shape
#load scambled image
mat_contents = sio.loadmat('Fmat.mat')
Fmat = mat_contents['Fmat']
[Hm, Wm] = Fmat.shape

F = Fmat.copy()
Fplt = np.log(abs(np.fft.fftshift(F))) #create log version to visualize  

mat = F.copy()
keys = permvec.astype(int).flatten()
ofstW = 160
ofstH = 160
bw = 20  
bh = 20
W = 4
H = 4

Fn = unscramble(mat, keys, ofstW, ofstH, bw, bh, W, H).copy()
#Fn = Fn[160:240,160:240]

Fnplt = np.log(abs(np.fft.fftshift(Fn))) #create log version to visualize
Fncenter = Fnplt[160:240,160:240] #get center file for submit
np.savetxt('A4.dat', Fncenter)

Fnshow = np.fft.ifft2(Fn) #inverse fft the matrix and return to real numbers
Fnshow = np.uint8(abs(Fnshow)) #convert numbers to 8 bit color values
np.savetxt('A5.dat', Fnshow) #reconstruct image matrix for submittal