
# coding: utf-8

# In[44]:


from __future__ import division
import numpy as np
import scipy
from scipy import integrate
from numpy import linalg as LA

import pdb
import matplotlib.pyplot as plt


# In[45]:


#import custom functions
#harmonic is assignment function
#sol_frame indexes array same as matlab

from harmonic import *
from sol_frame import *


# In[343]:


#set up frame

#BC values for x
x0=-4
xN=4

delta_x = 0.1
x_frame = sol_frame(x0, xN, delta_x)
K = 1


# In[376]:


#set up matrix A
a_dim = (len(x_frame), len(x_frame))
A = np.zeros(a_dim)


# In[377]:


#add middle terms to matrix for central difference
for i in range(1, len(A)-1):
    #calculate x value at i
    x = x_frame[i]

    A[i:(i+1),(i-1):i] = -1
    A[i:(i+1),i:(i+1)] = 2+K*(x**2)*delta_x**2
    A[i:(i+1),(i+1):(i+2)] = -1


# In[400]:


#Set up forward difference in first row

#set BC first row
A[0:1,0:1] = -3
A[0:1,1:2] = 4
A[0:1,2:3] = -1
#A[0:1,3:4] = 1

#set BC last row
A[a-1:a,a-1:a] = -3
A[a-1:a,a-2:a-1] = 4
A[a-1:a,a-3:a-2] = -1
#A[a-1:a,a-4:a-3] = 1


# In[401]:


[val, vect] = LA.eig(A)


# In[412]:


#save eigen values
A12 = np.r_[val[0], val[1], val[2], val[3], val[4]]
np.savetxt('A12.dat', A12)


# In[415]:


#save eigen vectors
A7 = vect[0]
np.savetxt('A7.dat', A7)

#save eigen vectors
A8 = vect[1]
np.savetxt('A8.dat', A8)

#save eigen vectors
A9 = vect[2]
np.savetxt('A9.dat', A9)

#save eigen vectors
A10 = vect[3]
np.savetxt('A10.dat', A10)

#save eigen vectors
A11 = vect[4]
np.savetxt('A11.dat', A11)


# #### Plots 
