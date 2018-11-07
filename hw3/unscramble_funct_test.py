# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:31:18 2018

@author: sspie

function test
"""
'''
import numpy as np
import scipy.io as sio
from unscramble import unscramble

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
[ofstW, ofstH] = [2,2] #offset from top left
[W,H] = [4,4] #block grid
grd = np.transpose(np.arange(1,W*H+1,1).reshape(W,H))
key = np.transpose(keys.reshape(W,H))
mat = h

Fn = unscramble(mat, keys, ofstW, ofstH, bw, bh, W, H)
'''

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from unscramble import unscramble

#load key
mat_contents = sio.loadmat('permvec.mat')
permvec = mat_contents['permvec']
[Hp, Wp] = permvec.shape
#load scambled image
mat_contents = sio.loadmat('Fmat.mat')
Fmat = mat_contents['Fmat']
[Hm, Wm] = Fmat.shape

F = Fmat[0:4,0:4]

[bw, bh] = [1,1] #block size
[ofstW, ofstH] = [0,0] #offset from top left
[W,H] = [4,4] #block grid
grd = np.transpose(np.arange(1,W*H+1,1).reshape(W,H))
keys = permvec.astype(int).flatten()
mat = F.copy()

Fn = unscramble(mat, keys, ofstW, ofstH, bw, bh, W, H)
















