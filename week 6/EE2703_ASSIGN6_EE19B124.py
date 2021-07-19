"""
        EE2703 Applied Programming Lab - 2021
            Assignment 6 solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE2703_ASSIGN6_EE19B124
            commandline INPUT : <filename.py> or <filename.py> <n> <M> <nk> <u0> <p>
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd    # pandas for showing in tabular form
import sys
from sys import argv, exit
#Default Values
n=100   # spatial grid size.
M=10    # number of electrons injected per turn.
nk=500  # number of turns to simulate.
u0=7    # threshold velocity.
p=0.5   # probability that ionization will occur
temp = 1
if(len(sys.argv)>1):
    if len(argv)!=6:
        print('Enter the correct number of values')
        print("The default values are being used")
    if len(argv)==6:
        for i in range(1,6):
            if float(argv[i]) <=0:
                print("Enter positive Values")
                print("The default values are being used")
                temp = 0
        if float(argv[5])>1:
            print("Probability should be less than 1")
            print("The default values are being used")
            temp = -1
        if temp == 1:
            n=int(argv[1])
            M=int(argv[2])
            nk=int(argv[3])
            u0=int(argv[4])
            p=float(argv[5])
#Initialising The vectors,lists
xx = np.zeros(n*M)  # electron position at each turn
u = np.zeros(n*M)   # electron velocity at each turn
dx = np.zeros(n*M)  # electron displacement at each turn

I = [] #Intensity of emission light
X = [] #Electron density
V = [] #Velocity Distribution

temp = np.where(xx>0)
ii = temp[0].tolist()
Msig = 2 #Standard Deviation
for i in range(nk):
    #Updating the position and velocity according to acceleration
    dx[ii] = u[ii] + 0.5
    xx[ii] = xx[ii] + dx[ii]
    u[ii] = u[ii] + 1
    #Resets the electron parameters Reached Anode
    Anode = np.where(xx>n)[0]
    xx[Anode] = 0
    u[Anode] = 0
    dx[Anode] = 0
    #Finding the electrons with velocity greater than Threshold
    kk = np.where( u >= u0 )[0]
    ll = np.where(np.random.rand(len(kk)) <= p)#Electron suffered Collision
    kl = kk[ll]
    #Set x,u after Collision
    u[kl] = 0
    xx[kl] = xx[kl] - dx[kl]*np.random.rand(1)[0]
    #Extending the intensity List with current emission position
    I.extend(xx[kl].tolist())
    #Injecting the new electrons into empty slots
    m = np.random.randn()*Msig + M
    Empty = (np.where(xx==0)[0])
    Occupy = min(len(Empty),(int)(m))
    xx[Empty[0:Occupy]] = 1.0
    u[Empty[0:Occupy]] = 0.0
    ii = np.where(xx > 0)[0]
    #Extending the X,V list with Current position of Electeons
    X.extend(xx[ii].tolist())
    V.extend(u[ii].tolist())
#Graphs:
plt.figure(1,figsize=(10,10))
plt.hist(I,histtype='bar',bins=n,ec='black',color = 'w',alpha=0.5)
#bins_1 = plt.hist(I,histtype='bar',bins=n,ec='black',color = 'w',alpha=0.5)[1]
plt.ylabel("Intensity")
plt.xlabel("xpos")
plt.title("Emission Intensity")

plt.figure(2,figsize=(10,10))
plt.hist(X,histtype='bar',bins=n,ec='black',color = 'w',alpha=0.5)[2]
plt.ylabel("Count")
plt.xlabel("xpos")
plt.title("Electron Density")

plt.figure(3,figsize=(10,10))
plt.plot(X,V,'x', markerfacecolor='blue', markersize=5)
plt.ylabel("Velocity")
plt.xlabel("Position")
plt.title("Electron phase space")
plt.show()
plt.figure()
bins = plt.hist(I,bins=np.arange(1,n,1))[1]
count = plt.hist(I,bins=np.arange(1,n,1))[0]
xpos = 0.5*(bins[0:-1] + bins[1:])
#Printing the Intensity data
f = pd.DataFrame(np.zeros((len(xpos),2)),columns = ['xpos','count'])
f = f.astype('object') #changing the datatype of the dataframe(to store lists)
f['xpos'] = xpos
f['count'] = count
print("Intensiy Data:")
print(f)