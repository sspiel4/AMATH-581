
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np


# In[2]:


#define error list
E = []
#set initial power to 2
p=2
#y_0 initial condition
y_0 = (np.pi / np.sqrt(2))
#norm list var
norm_list = []
#delta_t list
delta_t_list = []

for p in range(2,9):
    #define delta_T
    delta_t = 2**-p
    delta_t_list.append(delta_t)
    
    #define time interval
    start = 0 
    stop = 5

    #determine number of points in time array
    t_pts = int(1 + ((stop - start) / delta_t))
    #create numpy array for time interval
    t = np.linspace(start,stop,t_pts)

    #initiate y array
    y = []

    #Euler Method Calculation
    y.append(y_0)

    #loop through and calculate y values with euler
    for i in range(0,len(t)-1):
        #calculate dy/dt
        dydt = -3*y[i]*np.sin(t[i])
        y.append(y[i]+delta_t*dydt)
        
    #Real function calculation
    real = []
    for i in range(0,len(t)):
        real.append((np.pi*np.e**(3*(np.cos(t[i])-1))/ np.sqrt(2)))
        
        
    #reset norm variable
    norm = 0
    #loop through and calculate error
    for i in range(0,len(t)-1):
        norm = norm + ((real[i] - y[i])**2)
    
    norm = np.sqrt(norm)
    norm_list.append(norm)
    
#solutions
A1 = y
np.savetxt('A1.dat', A1)
A2 = norm_list
np.savetxt('A2.dat', A2)

#calculate polyfit line
slope, intercept = np.polyfit(np.log(delta_t_list),np.log(norm_list),1)
A3=[]
A3.append(slope)
np.savetxt('A3.dat', A3)

