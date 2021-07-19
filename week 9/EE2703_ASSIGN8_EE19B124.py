"""
        EE2703 Applied Programming Lab - 2021
            Assignment 8 solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE2703_ASSIGN8_EE19B124
            commandline INPUT : <FILE NAME>
"""
#Importing The Library

import pylab
from pylab import *

# Q1

#DFT of sin(5*x)
x=linspace(0,2*pi,129);x=x[:-1]
y=sin(5*x)

Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)
#Magnitude
figure(1,figsize=(10,10))
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)
#Phase
subplot(2,1,2)
plot(w,angle(Y),'+')
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'ro',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$k$",size=16)
grid(True)

#DFT of AM Modulated Signal with small time axis

t=linspace(0,2*pi,129);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)

figure(2,figsize=(10,10))
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'+')
xlim([-15,15])
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'ro',lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#DFT of AM Modulated Signal with extended time axis
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/512.0
w=linspace(-64,63,512)

figure(3,figsize=(10,10))
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'+')
xlim([-15,15])
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'ro',lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#Q2 

#DFT of sin^3(t)

t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(sin(t))**3
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]

figure(4,figsize=(10,10))
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin^3(t)$")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'+')
xlim([-15,15])
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'ro',lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#DFT of cos^3(t)

t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(cos(t))**3
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]

figure(5,figsize=(10,10))
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos^3(t)$")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'+')
xlim([-15,15])
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'ro',lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#Q3
#DFT of cos(20t+5cos(t))

t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(cos((20*t)+5*cos(t)))
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w = w[:-1]

figure(6,figsize=(10,10))
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-40,40])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $(\cos(20t + 5cos(t)) $")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'+',markersize = 1)
xlim([-40,40])
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'ro',lw=2.5)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#Q4 
#Spectrum of Guassian expression exp(-t^2/2)
"""
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y = exp(-t**2/2)
Y = fftshift(abs(fft(y)))/512
Y = Y * sqrt(2 * pi)/max(Y)        #Normalising to match both maximas
w=linspace(-64,64,513);w = w[:-1]"""
t=linspace(-4*pi,4*pi,2049);t=t[:-1]
y = exp(-t**2/2)
Y = fftshift(abs(fft(y)))/2048
Y = Y * sqrt(2 * pi)/max(Y)        #Normalising to match both maximas
w=linspace(-256,256,2049);w = w[:-1]
Y_ = exp(-w**2/2) * sqrt(2 * pi)  #True CTFT of Guassian Expression 

figure(7,figsize=(10,10))
subplot(2,1,1)
plot(w,abs(Y),lw=2,label = 'FFT')
plot(w,abs(Y_),'o',label = 'True')
xlim([-5,5])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of ${e^{-t^2}}/2 $")
legend()
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'+',markersize = 1)
xlim([-10,10])
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'ro',lw=2.5)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()