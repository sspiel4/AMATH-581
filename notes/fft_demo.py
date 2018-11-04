# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 20:56:07 2018

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

u = np.exp(-x*x) #function to take derivative
ut = np.fft.fft(u) #FFT the function

utshift = np.fft.fftshift(ut) #shift FFT

plt.figure(1), plt.plot(x, u)
plt.figure(2), plt.plot(x, abs(ut))
plt.figure(3), plt.plot(x, abs(utshift))

