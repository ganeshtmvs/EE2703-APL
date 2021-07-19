"""
        EE2703 Applied Programming Lab - 2021
            Assignment 6B solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE2703_ASSIGN6B_EE19B124
            commandline INPUT : <FILE NAME>
"""
#Importing the required Libraries

import numpy as np
import pylab
from pylab import *
import scipy
import scipy.signal as sp

#print( np.polymul([1,1,2.5],[1,0,2.25]))
#Q1

X = sp.lti([1,0.5],[1,1,4.75,2.25,5.625])#X(s) for f(t) = cos(1.5t)(e^(-0.5t))u(t)
t  = np.arange(0,50,0.01)
t,x=sp.impulse(X,0,t)#impulse response
#Graph:
figure(1,figsize=(8,6))
plot(t,x,label = 'cos(1.5t)e^{-0.5t}')

#Q2: Solving the above problem with smaller decay
#print(np.polymul([1,0.1,2.2525],[1,0,2.25]))

X = sp.lti([1,0.05],[1,0.1,4.5025,0.225,5.068125])#X(s) for f(t) = cos(1.5t)(e^(-0.05t))u(t)
t,x=sp.impulse(X,0,t) #Impulse Response as inverse laplace 

plot(t,x,label = 'cos(1.5t)e^{-0.05t}')
xlabel('Time')
ylabel('Position')
title('Position vs time plot')
legend()
grid()

#Q3: Varying the frequency of f(t) between 1.4 and 1.6

H = sp.lti([1],[1,0,2.25])
w  = np.arange(1.4,1.6,0.05)
t  = np.arange(0,150,0.1)

figure(2,figsize=(10,15))
k = 1
for i in range(5):
    subplot(3,2,k)
    #figure()
    f = np.cos(w[i]*t)*np.exp(-0.05*t)  #f(t) with chnaging frequency
    t,y,svec=sp.lsim(H,f,t) 
    
    plot(t,y)
    xlabel('Time')
    ylabel('Position')
    title('$\omega$ = ' + str(w[i]))
    grid() 
    if k == 3:
        k = k +1
    k = k+1
#show()
suptitle('Position vs time plot for various $\omega$')

#Q4 : Coupled Spring

X = sp.lti([1,0,2],[1,0,3,0]) #X(s)             
t,x = sp.impulse(X,None,T=np.linspace(0,20,1001))  #x(t)
Y = sp.lti([2],[1,0,3,0])     #Y(s)
t,y = sp.impulse(Y,None,T=np.linspace(0,20,1001))  #y(t)

figure(3,figsize=(8,8))
plot(t,x,label = 'x(t)')
plot(t,y,label = 'y(t)')
xlabel('Time')
ylabel('Position')
title('Position vs time plot of coupled springs')
legend()
grid()

#Q5 : Magnitude and Phase Response of Transfer function

H = sp.lti([1],[10**(-12),10**(-4),1]) #H(s) Transfer function
figure(4,figsize=(10,5))
w,S,phi=H.bode()

subplot(2,1,1)
semilogx(w,S,label = "Magnitude") #Magnitude plot
ylabel('log of magnitude')
title('Bode Plots of Transfer Function')
grid()
legend()

subplot(2,1,2)
semilogx(w,phi,label = "phase") #phase plot
xlabel('$\omega$')
ylabel('Phase')
grid()
legend()

#Q6 : input and output Voltages in time domain

#Smaller time range to capture the higer frequency effect
t = np.arange(0,30*(10**-6),0.01*10**-6)  
v_in = np.cos((10**3)*t) - np.cos((10**6)*t) #input with a high,low frequencies 
t,v_out,svec = sp.lsim(H,v_in,t)   #H(t)*Vin(t) 

figure(5,figsize=(10,15))
subplot(2,1,1)
plot(t,v_in,label = "Input")
ylabel('Voltage')
xlabel('Time(0 to 30 \u03bcs)')
title('Voltage Plot of input and output')
grid()
legend()

subplot(2,1,2)
plot(t,v_out,label = "Output")
ylabel('Voltage')
xlabel('Time(0 to 30 \u03bcs)')
#title('Voltage Plot of output')
grid()
legend()
#larger time range to capture the  frequency effect
t = np.arange(0,10*(10**-3),0.5*10**-6)  
v_in = np.cos((10**3)*t) - np.cos((10**6)*t) #input with a high,low frequencies 
t,v_out,svec = sp.lsim(H,v_in,t)   #H(t)*Vin(t)

figure(6,figsize=(10,15))
subplot(2,1,1)
plot(t,v_in,label = "Input")
ylabel('Voltage')
xlabel('Time(0 to 10ms)')
title('Voltage Plot of input and output')
grid()
legend()

subplot(2,1,2)
plot(t,v_out,label = "Output")
ylabel('Voltage')
xlabel('Time(0 to 10ms )')
#title('Voltage Plot of output')
grid()
legend()

show()