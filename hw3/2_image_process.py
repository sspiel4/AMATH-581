# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:20:50 2018

@author: sspie
"""

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

#F = np.fft.fftshift(Fmat) #shift Fmat so larger values in center
F = Fmat.copy()

Fplt = np.log(abs(np.fft.fftshift(F))) #create log version to visualize  
#Fplt = Fplt[160:240,160:240]
plt.imshow(Fplt, cmap=plt.cm.gray); plt.show() #plot log version

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
plt.imshow(Fnplt, cmap=plt.cm.gray); plt.show() #plot log version

#check original image
F = np.fft.ifft2(Fmat) #inverse fft the matrix and return to real numbers
F = np.uint8(abs(F)) #convert numbers to 8 bit color values
plt.imshow(F), plt.show() #show original

#check original image
Fnshow = np.fft.ifft2(Fn) #inverse fft the matrix and return to real numbers
Fnshow = np.uint8(abs(Fnshow)) #convert numbers to 8 bit color values
np.savetxt('A5.dat', Fnshow) #reconstruct image matrix for submittal
plt.imshow(Fnshow), plt.show() #show original



