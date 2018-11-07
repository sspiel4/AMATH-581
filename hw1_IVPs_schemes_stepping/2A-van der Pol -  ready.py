
# coding: utf-8

# In[1]:


import numpy as np
import scipy
from scipy import integrate
from numpy import linalg as LA


# ### Part A

# In[4]:


#establish a time range to compute the solution
start = 0
stop = 32
delta_t = 0.5
a_t = np.arange(start, stop, delta_t)

#initial values
init = [np.sqrt(3), 1]

#set up function
def van_der_Pol(Y, t):
    return [Y[1], -e*(Y[0]**2 - 1)*Y[1]-Y[0]]

#######################################################################################################################

#set e =0.1
e=0.1

asol = integrate.odeint(van_der_Pol, init, a_t)
y1a = asol[:,0:1]
y2a = asol[:,1:2]
       
#######################################################################################################################

#set e =1
e=1

#matrix returned from solvr, initial conditions, time range.
asol = integrate.odeint(van_der_Pol, init, a_t)
y1b = asol[:,0:1]
y2b = asol[:,1:2]
    
#######################################################################################################################

#set e=20
e=20

#matrix returned from solvr, initial conditions, time range.
asol = integrate.odeint(van_der_Pol, init, a_t)
y1c = asol[:,0:1]
y2c = asol[:,1:2]

A7 = np.c_[y1a, y1b, y1c]
np.savetxt('A7.dat', A7)

