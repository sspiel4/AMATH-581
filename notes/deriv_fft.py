# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 21:25:07 2018

@author: sspie
"""

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

L = 20 #define computational domain
n = 128 #define the number of fourier modes

x2 = np.linspace(-L/2, L/2, n+1) #define the domain discritization
x = x2[0:n] #consider only the first n points: periodic

u = 1 / np.cosh(x) #function to take derivative
ut = np.fft.fft(u) #FFT the function

k = (2*np.pi/L)*np.append(np.arange(0,(n/2),1), np.flip(np.arange(-1,((-n-1)/2),-1), 0)) #k rescale to 2pi domain

ut1 = (1j*k)*ut #take first dericative
ut2 = (-k**2)*ut #take second derivative
ut3 = (-1j*k**3)*ut #take third derivative

u1=np.fft.ifft(ut1) #inver transform
u2=np.fft.ifft(ut2) #inver transform
u3=np.fft.ifft(ut3) #inver transform

u1exact=-(1/np.cosh(x))*np.tanh(x)

utshift = np.fft.fftshift(ut) #shift FFT

plt.figure(1), plt.plot(x,u,'r',x,u1,'g',x,u1exact,'go',x,u2,'b',x,u3,'c')
