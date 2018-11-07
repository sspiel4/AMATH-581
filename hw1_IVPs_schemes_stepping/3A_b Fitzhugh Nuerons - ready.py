
# coding: utf-8

# In[1]:


import numpy as np
import scipy
from scipy import integrate
from numpy import linalg as LA


# In[2]:


#define fitzugh


# In[3]:


def coupled_fitzhugh(var, t, param):
    '''
    var : vector of variables
        vars = [v1, w1, v2, w2]
        
    t : time
        
    param : vector of parameters
        param = [a1, b1, c1, I1, d21, a2, b2, c2, I2, d12]
    '''
    
    v1, w1, v2, w2 = var
    a1, b1, c1, I1, d21, a2, b2, c2, I2, d12 = param
    
    dv1 = -v1**3+(1+a1)*v1**2-a1*v1-w1+I1+d12*v2
    dw1 = b1*v1
    
    dv2 = -v2**3+(1+a2)*v2**2-a2*v2-w2+I2+d21*v1
    dw2 = b2*v2
    
    return np.array([dv1,dw1,dv2,dw2])


# In[4]:


#solu A11


# In[5]:


#define constants
a1=0.05
b1=0.01
c1=0.01
I1=0.1
d12=0

a2=0.25
b2=0.01
c2=0.01
I2=0.1
d21=0

#define intit cond
v1=2
w1=0
v2=2
w2=0

# ODE solver parameters
abserr = 0.0025*1.0e-1  #to minick ode15 
relerr = 0.0025*1.0e-5  #to minick ode15
stoptime = 0.5
numpoints = 201

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
p = [a1, b1, c1, I1, d21, a2, b2, c2, I2, d12]
w0 = [v1, w1, v2, w2]

# Call the ODE solver.
wsol = integrate.odeint(coupled_fitzhugh, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

#write solution array
A11 = np.c_[wsol[:,0:1],wsol[:,2:3],wsol[:,1:2],wsol[:,3:4]]
np.savetxt('A11.dat', A11)


# In[6]:


#solu A12


# In[7]:


#define constants
a1=0.05
b1=0.01
c1=0.01
I1=0.1
d12=0

a2=0.25
b2=0.01
c2=0.01
I2=0.1
d21=0.2

#define intit cond
v1=2
w1=0
v2=2
w2=0

# ODE solver parameters
abserr = 0.0025*1.0e-1  #to minick ode15 
relerr = 0.0025*1.0e-5  #to minick ode15
stoptime = 0.5
numpoints = 201

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
p = [a1, b1, c1, I1, d21, a2, b2, c2, I2, d12]
w0 = [v1, w1, v2, w2]

# Call the ODE solver.
wsol = integrate.odeint(coupled_fitzhugh, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

#write solution array
A12 = np.c_[wsol[:,0:1],wsol[:,2:3],wsol[:,1:2],wsol[:,3:4]]
np.savetxt('A12.dat', A12)


# In[8]:


#solu A13


# In[9]:


#define constants
a1=0.05
b1=0.01
c1=0.01
I1=0.1
d12=-0.1

a2=0.25
b2=0.01
c2=0.01
I2=0.1
d21=0.2

#define intit cond
v1=2
w1=0
v2=2
w2=0

# ODE solver parameters
abserr = 0.0025*1.0e-1  #to minick ode15 
relerr = 0.0025*1.0e-5  #to minick ode15
stoptime = 0.5
numpoints = 201

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
p = [a1, b1, c1, I1, d21, a2, b2, c2, I2, d12]
w0 = [v1, w1, v2, w2]

# Call the ODE solver.
wsol = integrate.odeint(coupled_fitzhugh, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

#write solution array
A13 = np.c_[wsol[:,0:1],wsol[:,2:3],wsol[:,1:2],wsol[:,3:4]]
np.savetxt('A13.dat', A13)


# In[10]:


#solu A14


# In[11]:


#define constants
a1=0.05
b1=0.01
c1=0.01
I1=0.1
d12=-0.3

a2=0.25
b2=0.01
c2=0.01
I2=0.1
d21=0.2

#define intit cond
v1=2
w1=0
v2=2
w2=0

# ODE solver parameters
abserr = 0.0025*1.0e-1  #to minick ode15 
relerr = 0.0025*1.0e-5  #to minick ode15
stoptime = 0.5
numpoints = 201

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
p = [a1, b1, c1, I1, d21, a2, b2, c2, I2, d12]
w0 = [v1, w1, v2, w2]

# Call the ODE solver.
wsol = integrate.odeint(coupled_fitzhugh, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

#write solution array
A14 = np.c_[wsol[:,0:1],wsol[:,2:3],wsol[:,1:2],wsol[:,3:4]]
np.savetxt('A14.dat', A14)


# In[12]:


#solu A15


# In[13]:


#define constants
a1=0.05
b1=0.01
c1=0.01
I1=0.1
d12=-0.5

a2=0.25
b2=0.01
c2=0.01
I2=0.1
d21=0.2

#define intit cond
v1=2
w1=0
v2=2
w2=0

# ODE solver parameters
abserr = 0.0025*1.0e-1  #to minick ode15 
relerr = 0.0025*1.0e-5  #to minick ode15
stoptime = 0.5
numpoints = 201

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
p = [a1, b1, c1, I1, d21, a2, b2, c2, I2, d12]
w0 = [v1, w1, v2, w2]

# Call the ODE solver.
wsol = integrate.odeint(coupled_fitzhugh, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

#write solution array
A15 = np.c_[wsol[:,0:1],wsol[:,2:3],wsol[:,1:2],wsol[:,3:4]]
np.savetxt('A15.dat', A15)

