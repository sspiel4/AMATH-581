# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:44:15 2018

@author: sspie
"""

import numpy as np
import scipy.io as sio

h =  np.vstack(np.arange(1,8,1))
for i in range(2,10):
    v = np.vstack(np.arange(i,i+7,1))
    h = np.concatenate((h,v),axis=1)   
    
#load key
mat_contents = sio.loadmat('permvec.mat')
permvec = mat_contents['permvec']
[Hp, Wp] = permvec.shape
keys = permvec.astype(int).flatten()


[bw, bh] = [1,1] #block size
[ofstW, ofstH] = [0,0] #offset from top left
[W,H] = [4,4] #block grid
grd = np.transpose(np.arange(1,W*H+1,1).reshape(W,H))
key = np.transpose(keys.reshape(W,H))
h = grd

col = 1
row = 1
blocks = []

#for loop to gather blocks and put them in a list
for i in range(0,len(keys)):
    
    c = col -1 
    r = row - 1
    
    col_str = ofstW + bw*c 
    col_stp = ofstW + bw*c+bw 
    row_str = ofstH + bh*r
    row_stp = ofstH + bh*r+bh
    block = np.array(h[row_str:row_stp,col_str:col_stp], copy=True)
    blocks.append(block)
  
    if row == H:
        row = 0
        col = col + 1
        
    row = row + 1

col = 1 #reset columns
row = 1 #reset rows

#second, similiar, for loop places blocks in there new spot!
for i in range(0,len(keys)):
    #get fist block by keys index
    index = keys[i]-1
    block = blocks[index]
    
    c = col -1 
    r = row - 1
    
    col_str = ofstW + bw*c 
    col_stp = ofstW + bw*c+bw
    row_str = ofstH + bh*r
    row_stp = ofstH + bh*r+bh
    h[row_str:row_stp,col_str:col_stp] = block #replace section of array
  
    
    if row == H:
        row = 0
        col = col + 1
        
    row = row + 1
    

    
    