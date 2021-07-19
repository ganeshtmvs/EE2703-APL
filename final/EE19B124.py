"""
        EE2703 Applied Programming Lab - 2021
            Final Exam
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE19B124.py
            commandline INPUT : <FILE NAME>
"""
#Importing the Required Libraries
import numpy as np
import pylab
from pylab import *

#Q2 - Divinding the Volume into a 3 X 3 X 1000 Meshpoints 

i = np.array([0,1,2])                                                                    #X-Coordinates
j = np.array([0,1,2])                                                                    #Y-Coordinates
k = np.linspace(1,1000,1000)                                                             #Z-Coordinates from  1 to 1000cm with a step of 1cm.
xx,yy,zz= meshgrid(i,j,k)                                                                #Meshgrid 
            
#Q3- Dividing the current carrying loop into N Sections     

r = 10                                                                                   #Radius of loop(r=a=k)
N = 100                                                                                  #No.of Sections 
phi = np.linspace(0,2*np.pi,N)                                                           #Ploar angle array at the corresponding Section 
x = np.array(r*(np.cos(phi)))                                                            #X-Coordinates of loop(r*cos(phi))
y = np.array(r*(np.sin(phi)))                                                            #Y-Coordinates of loop(r*sin(phi))

#Q5- r_l and dl vectors 

r_l = c_[x,y,np.zeros(N)]                                                                #3D vector positions of the sections in loop
dl  = c_[-y,x,np.zeros(N)]                                                               #Tangential Vector at Sections of loop(-rsin(phi)i+rcos(phi)j^)

#Q4- plot of Current Flowing in the Loop 

u = 4*np.pi*(10**-7)                                                                     #permeability of free space 
I_x = ((4*np.pi)/u)*np.cos(phi)*dl.T[0]                                                  #X - component of Current in the loop
I_y = ((4*np.pi)/u)*np.cos(phi)*dl.T[1]                                                  #Y - component of Current in the loop

figure("Current in the loop",figsize = (8,8))                                            #Figure 
quiver(x, y, I_x, I_y,headlength=5,headwidth = 5,width=0.003,pivot = 'mid')              #Quiver plot of Current in loop
#colorbar()
xlabel("X-coordinates",fontsize = 15)
ylabel("Y-Cordinates",fontsize = 15)
title("Current Vector in the loop",fontsize = 15)
xticks(np.arange(-10, 11, step=1))
yticks(np.arange(-10, 11, step=1))
grid()

#Q6,7 Defining the calc function to calculate R_ijkl and extending it to find A vector

def calc(l):                                                                             #function to calculate R,A matrices 
    R_ijkl = np.sqrt(((xx-r_l[l][0])**2)+((yy-r_l[l][1])**2)+(zz**2))                    #The value of R_ijkl all over the 3X3X1000 meshgrid for a particular l
    A_ijkl_x = (np.cos(phi[l])*(dl[l][0]) * (1/R_ijkl))* (np.exp(-1j*0.1*R_ijkl))        #The x component of potential all over mesh grid for a particular l
    A_ijkl_y = (np.cos(phi[l])*(dl[l][1]) * (1/R_ijkl))* (np.exp(-1j*0.1*R_ijkl))        #The y component of potential all over mesh grid for a particular l
    return A_ijkl_x,A_ijkl_y,R_ijkl

def potential():                                                                         #Function to calculate the potential Vector                                                                        
    A_x = calc(0)[0]                                                                     #Initialing A_x 3X3X1000 matrix with l= 0 from calc fn
    A_y = calc(0)[1]                                                                     #Initialing A_y 3X3X1000 matrix with l= 0 from calc fn
    for l in range(1,N):
        A_x += calc(l)[0]                                                                #X component of potential for all (i,j,l)
        A_y += calc(l)[1]                                                                #X component of potential for all (i,j,l)
    return A_x,A_y    

A_x,A_y= potential()   

#Q8 - Calculating Z component of Magnetic field

B_z = (A_y[2,1,:]-A_x[1,2,:]-A_y[0,1,:]+A_x[1,0,:])/(4)                                  #Magnetic field according to equation 2 in the pdf 

#Q9 - loglog plot of the Magnetic field

figure("Magnetic Field ",figsize = (8,6))
loglog(k, np.abs(B_z),label = "B_z")                                                     #log(B) vs log(z) plot of magnetic field 
xlabel("Z-Cordinates(log)",fontsize = 15)
ylabel("Z comp of Magnetic field(B_z)",fontsize = 15)
title("Magnetic field along x=1,y=1,z",fontsize = 15)
legend(loc = 'upper right',fontsize = 15)
grid()
#show()

#Q10 - Fitting the data to a exponential using least squares method

M = c_[np.ones(1000),np.log(k)]                                                          #M*p = X ,M is a vector of 1, log(k)
X = c_[np.log(np.abs(B_z))]                                                              # X is vector of log(Magnetic field)
p = lstsq(M,X,rcond=None)
c = np.exp(p[0][0])
b = (p[0][1])
print("The Value of c and b are:")
print(c,b)
B_fit = c*(k**b)                                                                         #value of fitted data
figure("least square Fit",figsize = (8,6))                                               #Graph to compare both the plots
loglog(k, np.abs(B_z),label = "Original")
loglog(k, np.abs(B_fit),label = "fit")
xlabel("Z-Cordinates(log)",fontsize = 15)
ylabel("Z comp of Magnetic field(B_z)",fontsize = 15)
title("Magnetic field fit vs original",fontsize = 15)
legend(loc = 'upper right',fontsize = 15)
grid()
show()