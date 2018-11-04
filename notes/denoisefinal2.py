# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 07:36:29 2018

@author: sspie
"""

import numpy as np
from numpy import linalg as LA
import math

import time
import itertools as itertools

from PIL import Image
import scipy
from scipy import misc
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm

I = plt.imread('recorder.jpg') #read in image
plt.imshow(I), plt.show() #show original

[H, W, D] =  np.shape(I) # H = image height, W = image width, D = channels (red,green,blue)
newI = []


for i in range(0, 3):
    
    F = np.fft.fft2(I[:,:,i]) #apply fft2 to each of the three image channels
    F = np.fft.fftshift(F) #apply fft shift
    
    #################################
    
    #F1 = np.fft.ifftshift(F)       
    #F1 = np.fft.ifft2(F1)
    #F1 = np.uint8(abs(F1))
    #plt.imshow(F1), plt.show()
    
    ##################################
    
    
    #this array is used to zero out the values we don't want
    zeros = np.zeros((H,W)) #create array of zeros with height and width of image
    
    #how many outer values
    vert = (H/2)-10
    horz = (W/2)-60
    
    
    #set outer values to zeros array
    F[0:vert,:] = zeros[0:vert,:] #set top to zero
    F[H-vert:H,:] = zeros[H-vert:H,:] #set bottom to zero
    F[:,0:horz] = zeros[:,0:horz] #set first columns to zero
    F[:,W-horz:W] = zeros[:,W-horz:W] #set last columns to zero
    
    F = np.fft.ifftshift(F) #shift in freq domain back to original
    F = np.fft.ifft2(F) #inverse fft the matrix and return to real numbers
    F = np.uint8(abs(F)) #convert numbers to 8 bit color values
    
    
    newI.append(F) #create list of matrixes that will use to build 3d array

    
new_img = np.array([newI[0],newI[1],newI[2]]) #build 3d array (3, H, W)
new_img = np.swapaxes(new_img,0,2) #rearrange axis (W, H, 3)
new_img = np.swapaxes(new_img,0,1) #rearrange axis (H, W, 3)

plt.imshow(new_img), plt.show() #show filtered image
    
    

