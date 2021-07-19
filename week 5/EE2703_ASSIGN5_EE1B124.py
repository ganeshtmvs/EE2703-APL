"""
        EE2703 Applied Programming Lab - 2021
            Assignment 5 solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE2703_ASSIGN5_EE19B124
            commandline INPUT : <filename.py> or <filename.py> <Nx> <Ny> <radius> <Niter>
"""
#Importing the Required Libraries
import pylab
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import scipy
import scipy.special as sp

from sys import argv, exit
#Initialising the Parameters
Nx = 25;  #size along x
Ny = 25;  #size along y
radius = 8;#radius of central lead
Niter = 1500; # number of iterations to perform
temp = 1
if(len(sys.argv)>1):
    if len(argv)!=5:
        print('Enter the correct number of values')
        print("The default values are being used")
    if len(argv)==5:
        if argv[1]!=argv[2]:
            print("Enter equal values of Nx,Ny as we are using a square plate")
            print("The default values are being used")
            temp = 0
        if int(argv[3])>(int(argv[2])//2):
            print("Enter a smaller value of radius")
            print("The default values are being used")
            temp = -1
        if temp == 1:
            Nx=int(argv[1]); #size along x
            Ny=int(argv[2]); # size along y
            radius=int(argv[3]);#radius of central lead
            Niter=argv[4]; # number of iterations to perform
else:
    print("The default values are being used")

#Allocating and Initialising the potential array
phi = np.zeros((Nx,Ny))

x = arange(0,Nx,1)
x = x - (Nx//2)#Coordinates with (0,0) as midpoint
y = arange(0,Ny,1)
y = y - (Ny//2)
X,Y = meshgrid(x,y)

ii = where(X**2+Y**2 <= radius**2)#Indices of the 1V potential in phi array
phi[ii] = 1.0
#Graph:
figure(1)
contour(X,Y,phi,100,cmap=cm.jet)
colorbar()
scatter(X,Y, s = 1)
scatter(ii[0]-Ny//2,ii[1]-Nx//2,s = 2, color = 'r',label ='Current Carrying wire')#adjusting the indices ii to coordinates
xlabel('X(bottom side of plate)')
ylabel('Y(Left side of Plate)')
title('Initial Contour plot')
legend(loc = 'upper left',fontsize = 10)
#grid()
#Iterations:

def update_phi(phi): #function to update value of phi with average of neighbouring values
    phi[1:-1,1:-1] = 0.25*(phi[1:-1,0:-2]+phi[1:-1,2:]+phi[0:-2,1:-1]+phi[2:,1:-1])
def boundaries_phi(phi): #Asserting the Boundary conditions
    phi[1:-1,[0,-1]] = phi[1:-1,[1,-2]]
    phi[0] = phi[1]
    phi[ii] = 1
oldphi = phi.copy() #Initialisng the copy of phi
errors = np.zeros(Niter)#array of Errors
for i in range(Niter):#Iterating Niter no of times
    oldphi = phi.copy()
    update_phi(phi)
    boundaries_phi(phi)
    errors[i]=(abs(phi-oldphi)).max()

n_iter = arange(1,Niter+1,1)#array of no of iterations
figure(2)
plot(n_iter[::50],errors[::50],'o',label='Error')#Every 50th Value of Error
xlabel('iteations')
ylabel('Error')
title('Error in every iteration with a step of 50')
legend(loc = 'upper right',fontsize = 10)
figure(3)
loglog(n_iter,errors,label = 'Error')
loglog(n_iter[::50],errors[::50],'o',markersize =5)
xlabel('iteations(log)')
ylabel('Error(log)')
title('Error in every iteration with a step of 50 in log scale')
legend(loc = 'upper right',fontsize = 10)
show()
figure(4)
semilogy(n_iter,errors,label ="Error")
semilogy(n_iter[::50],errors[::50],'o',markersize =5)
xlabel('iteations')
ylabel('Error(log)')
title('Error in every iteration with a step of 50 in semilog axis')
legend(loc = 'upper right',fontsize = 10)
show()
#Finding the best fit to errors using least squares method
log_errors = np.log(errors)
M_1 = c_[ones(1500),n_iter]
X_1 = c_[log_errors]
p_1=lstsq(M_1,X_1,rcond=None)
fit1 = (exp(p_1[0][0]))*(exp(p_1[0][1]*n_iter))
#Finding the best fit to errors beyond 500 iterations using least squares method
M_2 = c_[ones(1500),n_iter]
X_2 = c_[log_errors]
p_2=lstsq(M_2,X_2,rcond=None)
fit2 = (exp(p_2[0][0]))*(exp(p_2[0][1]*n_iter))
#Graphs:
figure(5)
semilogy(n_iter,fit1,label = 'fit_1',linewidth = 3,color = 'b')
semilogy(n_iter,errors,label = 'original errors',linewidth = 2)
semilogy(n_iter,fit2,label = 'fit_2',linewidth = 1)
xlabel('iteations')
ylabel('Error(log)')
title('Error and fitted values in semilog plot')
legend(loc = 'upper right',fontsize = 10)
show()
#Stopping Condition
error_beyond_N= exp(p_1[0][0]+(p_1[0][1]*Niter))/(-p_1[0][1])#Maximum Error between the function from Niter to that of with infinite value

#Surface plot of the Potential
fig1=figure(6) # open a new figure
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y,-X, phi, rstride=1, cstride=1, cmap=cm.jet)

#Contour Plot of Potential
figure(7)
contour(X,-Y,phi,100,cmap=cm.jet)
colorbar()
scatter(ii[0]-Ny//2,ii[1]-Nx//2,s = 2, color = 'r')
xlabel('X(bottom side of Plate)')
ylabel('Y(Left side of Plate)')
title(' Contour plot of Potential after N iterations')
#legend()
grid()

#Currents:

J_x = np.zeros((Nx,Ny))#initialising the Values of Currents
J_y = np.zeros((Nx,Ny))

J_x[1:-1,1:-1] = 0.5*(phi[2:,1:-1] - phi[0:-2,1:-1])#Partial derivative of potential with sigma = 1
J_y[1:-1,1:-1] = 0.5*(phi[1:-1,2:] - phi[1:-1,0:-2])
#Vector Plolt of Currents
figure(8)
quiver(y,x,J_y[::-1,:],J_x[::-1,:],label = 'Current')
scatter(ii[0]-Ny//2,ii[1]-Nx//2,s = 2, color = 'r')
xlabel('X(bottom side of Plate)')
ylabel('Y(Left side of Plate)')
title('Current Flow inside the Plate')
legend(loc = 'upper right',fontsize = 10)
#Temperature:

Temperature = np.zeros((Nx,Ny))#initialising the Temperature array
ohmic = np.zeros((Nx,Ny))##initialising the ohmic loss array
Temperature = Temperature + 300 #initial Temperature
k = 1 #thermal Conductivity
sigma = 1
delta = 1
const  = (delta**2/(4*sigma*k))
def update_temp(Temperature):
    Temperature[1:-1,1:-1] = 0.25*(Temperature[1:-1,0:-2]+Temperature[1:-1,2:]+Temperature[0:-2,1:-1]+Temperature[2:,1:-1])
    ohmic = const*(J_x**2 + J_y**2)#Due to ohmic losses
    Temperature[1:-1,1:-1]  = Temperature[1:-1,1:-1] + ohmic[1:-1,1:-1]
def boundaries_temp(Temperature):
    Temperature[1:-1,[0,-1]] = Temperature[1:-1,[1,-2]]
    Temperature[0] = Temperature[1]
    Temperature[ii] = 300

for i in range(Niter):
    update_temp(Temperature)
    boundaries_temp(Temperature)
figure(9)
contourf(X,-Y,Temperature,cmap=cm.jet)
colorbar()
xlabel('X(bottom side of Plate)')
ylabel('Y(Left side of Plate)')
title('Temperature Contour')
#legend()