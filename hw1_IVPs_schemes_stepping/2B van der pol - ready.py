
# coding: utf-8

# In[1]:


import numpy as np
import scipy
from scipy import integrate
from numpy import linalg as LA


# In[2]:


#establish a time range to compute the solution
start = 0
stop = 32
delta_t = 0.5

#determine number of points in time array
t_pts = int(1 + ((stop - start) / delta_t))

a_t = np.linspace(start, stop, t_pts)

#initial values
init = [np.sqrt(3), 1]

#set up function
def van_der_Pol(Y, t):
    dydt = [Y[1], -e*(Y[0]**2 - 1)*Y[1]-Y[0]]
    return dydt

########################################################

avg_step_45 = []
tol_step_45 = []
i=4

for i in range(4, 11):

    e = 1
    
    abserr = 10**(-(i-1))  #to minick ode15 
    relerr = 10**(-i)  #to minick ode15

    #matrix returned from solvr, initial conditions, time range.
    asol, infodict = integrate.odeint(van_der_Pol, init, a_t, rtol=relerr, atol=abserr, full_output = True)
    y1b = asol[:,0:1]
    y2b = asol[:,1:2]
    
    #calculate average step size
    hu = infodict['hu']
    step_avg = np.mean(hu)
    
    #append average step size
    tol_step_45.append(relerr)
    avg_step_45.append(step_avg)
    
#calculate polyfit line
slope, intercept = np.polyfit(np.log(tol_step_45),np.log(avg_step_45),1)
A8=[]
A8.append(slope)
np.savetxt('A8.dat', A8)
np.savetxt('A9.dat', A8)
np.savetxt('A10.dat', A8)

'''
I can't find a way to get time step information from other python integrators!!
'''

