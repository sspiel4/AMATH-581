#calculations
from __future__ import division
import numpy as np
import scipy
from scipy import integrate
from numpy import linalg as LA
import pdb


#import custom functions
#harmonic is assignment function
#sol_frame indexes array same as matlab

from harmonic import *
from sol_frame import *


###############################################################


#define constants
tol = 10**-4
k = 1
L=4
xp = [-L,L]
A = 1 #initial derivative of function (a guess)

En_start = k
En = En_start

# Pack up the parameters and initial conditions:
p = [k, En]

#initial conditions
val = A
slope = A*np.sqrt(k*L**2 - En)
y0 = [val , slope]

#define x range for eig_funct
dx = 0.1
x_frame = sol_frame(xp[0], xp[1], dx)

eig_funct = []
eig_vals = []

for i in range(1,6):
    dEn = k/100
    En = En + dEn
    #loop through and find solution
    for j in range(1, 1000):

        #reset En
        p = [k, En]

        #reset initial conditions
        val = A
        slope = A*np.sqrt(k*L**2 - En)
        y0 = [val , slope]


        #solve ODE
        sol = integrate.odeint(harmonic, y0, x_frame, args=(p,))

        #pull end value
        end = len(sol)
        end_val = sol[end-1:end, 0:1]

        #check if solution is within tolerance
        if abs(end_val) < tol:
            En
            #print('found solution on iteration:')
            #print j
            break;

        if i % 2 == 0:
            if end_val < 0:
                En = En+dEn
            else:
                En = En-dEn/2
                dEn = dEn/2
        else:
            if end_val > 0:
                En = En+dEn
            else:
                En = En-dEn/2
                dEn = dEn/2

    y1a = sol[:,0:1]
    y2a = sol[:,1:2]

    #append eigen function to list
    eig_funct.append(y1a)
    #append Eigen value
    eig_vals.append(En)


# In[124]:


#write out solution of asbsolute value eigen funtions
A1 = abs(eig_funct[0])
A1 = A1 / np.trapz(A1, axis=-0, dx=dx)
np.savetxt('A1.dat', A1)

A2 = abs(eig_funct[1])
A2 = A2 / np.trapz(A2, axis=-0, dx=dx)
np.savetxt('A2.dat', A2)

A3 = abs(eig_funct[2])
A3 = A3 / np.trapz(A3, axis=-0, dx=dx)
np.savetxt('A3.dat', A3)

A4 = abs(eig_funct[3])
A4 = A4 / np.trapz(A4, axis=-0, dx=dx)
np.savetxt('A4.dat', A4)

A5 = abs(eig_funct[4])
A5 = A5 / np.trapz(A5, axis=-0, dx=dx)
np.savetxt('A5.dat', A5)


# In[125]:


eig1 = eig_vals[0] #/ np.trapz(A1, axis=-0, dx=dx)
eig2 = eig_vals[1] #/ np.trapz(A2, axis=-0, dx=dx)
eig3 = eig_vals[2] #/ np.trapz(A3, axis=-0, dx=dx)
eig4 = eig_vals[3] #/ np.trapz(A4, axis=-0, dx=dx)
eig5 = eig_vals[4] #/ np.trapz(A5, axis=-0, dx=dx)


# In[126]:


A6 = np.r_[eig1,eig2,eig3,eig4,eig5]
np.savetxt('A6.dat', A6)
A6


###############################################################

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


#######################################################################

from nonlinear_harmonic import *
from sol_frame import *


# In[15]:


#define constants
tol = 10**-4
k = 1
xp = [-2,2]
A_start = 1
A = A_start

En_start = k
En = En_start

#define x range for eig_funct
dx=0.1
x_frame = sol_frame(xp[0], xp[1], dx)

eig_funct = []
eig_vals = []


# In[ ]:


count = 0
gamma = 0.05

#loop through A
for i in range (1,10):
    dA = 0.1
    A = A + dA

    for i in range(1,2):
        dEn = k/100
        En = En + dEn
        #loop through and find solution
        for j in range(1, 1000):

            #reset En
            y0 = [0,A]
            p = [k, En, gamma]
            #solve ODE
            sol = integrate.odeint(nonlinear_harmonic, y0, x_frame, args=(p,))

            #pull end value
            end = len(sol)
            end_val = sol[end-1:end, 0:1]

            #check if solution is within tolerance
            if abs(end_val) < tol:
                En
                count = count + 1
                break;

            if i % 2 == 0:
                if end_val < 0:
                    En = En+dEn
                else:
                    En = En-dEn/2
                    dEn = dEn/2
            else:
                if end_val > 0:
                    En = En+dEn
                else:
                    En = En-dEn/2
                    dEn = dEn/2

        y1a = sol[:,0:1]
        y2a = sol[:,1:2]

        #append eigen function to list
        eig_funct.append(y1a)
        #append Eigen value
        eig_vals.append(En)

    if count > 1:
        break;

#write out solution of asbsolute value eigen funtions
A13 = abs(eig_funct[0])
A13 = A13 / np.trapz(A13, axis=-0, dx=dx)
np.savetxt('A13.dat', A13)

A14 = abs(eig_funct[1])
A14 = A14 / np.trapz(A14, axis=-0, dx=dx)
np.savetxt('A14.dat', A14)

eig1 = eig_vals[0] #/ np.trapz(A1, axis=-0, dx=dx)
eig2 = eig_vals[1] #/ np.trapz(A2, axis=-0, dx=dx)

A15 = np.r_[eig1,eig2]
np.savetxt('A15.dat', A15)


# In[ ]:


count = 0
gamma = -0.05

#loop through A
for i in range (1,10):
    dA = 0.1
    A = A + dA

    for i in range(1,2):
        dEn = k/100
        En = En + dEn
        #loop through and find solution
        for j in range(1, 1000):

            #reset En
            y0 = [0,A]
            p = [k, En, gamma]
            #solve ODE
            sol = integrate.odeint(nonlinear_harmonic, y0, x_frame, args=(p,))

            #pull end value
            end = len(sol)
            end_val = sol[end-1:end, 0:1]

            #check if solution is within tolerance
            if abs(end_val) < tol:
                En
                count = count + 1
                break;

            if i % 2 == 0:
                if end_val < 0:
                    En = En+dEn
                else:
                    En = En-dEn/2
                    dEn = dEn/2
            else:
                if end_val > 0:
                    En = En+dEn
                else:
                    En = En-dEn/2
                    dEn = dEn/2

        y1a = sol[:,0:1]
        y2a = sol[:,1:2]

        #append eigen function to list
        eig_funct.append(y1a)
        #append Eigen value
        eig_vals.append(En)

    if count > 1:
        break;

#write out solution of asbsolute value eigen funtions
A16 = abs(eig_funct[0])
A16 = A16 / np.trapz(A16, axis=-0, dx=dx)
np.savetxt('A16.dat', A16)

A17 = abs(eig_funct[1])
A17 = A17 / np.trapz(A17, axis=-0, dx=dx)
np.savetxt('A17.dat', A17)

eig1 = eig_vals[0]
eig2 = eig_vals[1]

A15 = np.r_[eig1,eig2]
np.savetxt('A18.dat', A18)
