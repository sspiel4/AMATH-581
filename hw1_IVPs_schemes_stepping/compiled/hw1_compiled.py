

import numpy as np
import scipy
from scipy import integrate
from numpy import linalg as LA


##### 1A Heun's #####

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
    E = 0
    #loop through and calculate error
    E = np.average(abs(real[i] - y[i]))
    norm_list.append(E)

#solutions
A1 = y
np.savetxt('A1.dat', A1)
A2 = norm_list
#A2 = np.asarray(A2)
#np.matrix.transpose(A2)
A2 = np.c_[A2[0],A2[1],A2[2],A2[3],A2[4],A2[5],A2[6]]
np.savetxt('A2.dat', A2)

#calculate polyfit line
slope, intercept = np.polyfit(np.log(delta_t_list),np.log(norm_list),1)
A3=[]
A3.append(slope)
np.savetxt('A3.dat', A3)

##### 1B Euler #####

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

    #loop through and calculate y values with heun's
    for i in range(0,len(t)-1):
        #calculate dy/dt
        dydt = -3*y[i]*np.sin(t[i])
        #calc dydt plus delta_t
        dydt_dt = -3*(y[i]+delta_t*dydt)*np.sin(t[i+1])
        y.append(y[i]+(delta_t/2)*(dydt+dydt_dt))

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
A4 = y
np.savetxt('A4.dat', A4)
A5 = norm_list
#A5 = np.asarray(A5)
#np.matrix.transpose(A5)
A5 = np.c_[A5[0],A5[1],A5[2],A5[3],A5[4],A5[5],A5[6]]
np.savetxt('A5.dat', A5)

#calculate polyfit line
slope, intercept = np.polyfit(np.log(delta_t_list),np.log(norm_list),1)
A6=[]
A6.append(slope)
np.savetxt('A6.dat', A6)


##### 2A Van der Pol #####

#establish a time range to compute the solution

#################
start = 0
stop = 32
delta_t = 0.5

#determine number of points in time array
t_pts = int(1 + ((stop - start) / delta_t))

a_t = np.linspace(start, stop, t_pts)

#####################

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


##### 2B Van der Pol #####

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

##### 3 fitzugh #####

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
