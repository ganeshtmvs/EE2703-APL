"""
        EE2703 Applied Programming Lab - 2021
            Assignment 9 solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE2703_ASSIGN9_EE19B124
"""

#Importing The libraries
from pylab import *
from mpl_toolkits.mplot3d.axes3d import Axes3D

def DFT_Plot(N,limits,function,xlimits,title_):
    t = linspace(limits[0],limits[1],N+1);t=t[:-1]
    dt = t[1]-t[0];fmax=1/dt
    y=function(t)
    n=arange(N)
    y[0]=0
    y=fftshift(y)
    Y=fftshift(fft(y))/N
    w=linspace(-pi*fmax,pi*fmax,N+1);w=w[:-1]
    figure(figsize = (10,10))
    subplot(2,1,1)
    plot(w,abs(Y),'b',w,abs(Y),'bo',lw=1.5,markersize = 2.5)
    xlim([xlimits[0],xlimits[1]])
    ylabel(r"$|Y|$",size=16)
    title(title_)
    grid(True)
    subplot(2,1,2)
    plot(w,angle(Y),'+')
    ii=where(abs(Y)>1e-3)
    plot(w[ii],angle(Y[ii]),'ro',lw=2)
    #plot(w,angle(Y),'ro',lw=2)
    xlim([xlimits[0],xlimits[1]])
    ylabel(r"Phase of $Y$",size=16)
    xlabel(r"$\omega$",size=16)
    grid(True)
    #show()
    return w,Y

def DFT_Plot_Wind(N,limits,function,xlimits,title_):
    t = linspace(limits[0],limits[1],N+1);t=t[:-1]
    dt = t[1]-t[0];fmax=1/dt
    y=function(t)
    n=arange(N)
    wnd=fftshift(0.54+0.46*cos(2*pi*n/(N)))
    y = y*wnd
    y[0]=0
    y=fftshift(y)
    Y=fftshift(fft(y))/N
    w=linspace(-pi*fmax,pi*fmax,N+1);w=w[:-1]
    figure(figsize=(10,10))
    subplot(2,1,1)
    plot(w,abs(Y),'b',w,abs(Y),'bo',lw=1.5,markersize = 2.5)
    xlim([xlimits[0],xlimits[1]])
    ylabel(r"$|Y|$",size=16)
    title(title_)
    grid(True)
    subplot(2,1,2)
    #plot(w,angle(Y),'ro',lw=2)
    plot(w,angle(Y),'+')
    ii=where(abs(Y)>1e-3)
    plot(w[ii],angle(Y[ii]),'ro',lw=1)
    xlim([xlimits[0],xlimits[1]])
    ylabel(r"Phase of $Y$",size=16)
    xlabel(r"$\omega$",size=16)
    grid(True)
    #show()
    return w,Y

#Q2 - spectrum of cos^3(wt)

def cos_3(t,w_o=0.86):
    return cos(w_o*t)**3

DFT_Plot(256,[-4*pi,4*pi],cos_3,[-4,4],r"Spectrum of $\cos^3\left(w_ot\right)$ without windowing")
DFT_Plot_Wind(256,[-4*pi,4*pi],cos_3,[-4,4],r"Spectrum of $\cos^3\left(w_ot\right)$ with windowing")

#Q3 - Spectrum of cosine with phase

def cos_delta(t,w0 = 0.86,delta =0):
    print("The true Values are : %.3f and %.3f "%(w0,delta))
    #print( )
    return cos(w0*t + delta)

def calculate_w(w,Y):
    ii = where(abs(Y)>0.2)
    w_estimated = sum((abs(Y[ii])**2) * abs(w[ii]))/sum(abs(Y[ii])**2)
    jj = where(abs(Y)==max(abs(Y)))
    phase_estimated = mean(abs(angle(Y[jj])))
    return w_estimated,phase_estimated

p,k = DFT_Plot(128,[-4*pi,4*pi],cos_delta,[-6,6],r"Spectrum of $\cos(\omega_o*t + \delta)$ without windowing")
print(r"The estimated value of w0 and phase are ")
print(calculate_w(p,k))
print()
p,k =DFT_Plot_Wind(128,[-4*pi,4*pi],cos_delta,[-6,6],r"Spectrum of $\cos(w_o*t + \delta)$ with windowing")
print(r"The estimated value of w0 and phase with windowing are")
print(calculate_w(p,k))
print()

#Q4 - Spectrum of cosine with phase and random noise

def cos_noise(t,w0 = 0.86 ,delta = 0,noise = 0.1):
    print("The true Values (with noise %.3f) are : %.3f and %.3f:" %(noise,w0,delta))
    #print( )
    return cos(w0*t + delta) + noise*randn(len(t))

p,k = DFT_Plot(128,[-4*pi,4*pi],cos_noise,[-6,6],r"Spectrum of $\cos(w_o*t + \delta) + noise$ without windowing")
print(r"The estimated value of w0 and phase are")
print(calculate_w(p,k))
print()
p,k =DFT_Plot_Wind(128,[-4*pi,4*pi],cos_noise,[-6,6],r"Spectrum of $\cos(w_o*t + \delta) + noise$ with windowing")
print(r"The estimated value of w0 and phase with windowing are")
print(calculate_w(p,k))


#Q5 - Chirped Signal

def chirp(t):
    return cos(16*(1.5+t/(2*pi))*t)

DFT_Plot(1024,[-pi,pi],chirp,[-60,60],r"Spectrum of $cos(16(1.5+\frac{t}{2\pi})t)$ without windowing")
DFT_Plot_Wind(1024,[-pi,pi],chirp,[-60,60],r"Spectrum of $cos(16(1.5+\frac{t}{2\pi})t)$ with windowing")

#Q6 - Chirped Signal with breaking time 

chirped_=zeros([64,16],dtype=complex)
for i in range(0,16):
    t=linspace(-pi+(i/8)*pi,-pi+((i+1)/8)*pi,65);t=t[:-1]
    y=chirp(t)
    n=arange(64)
    wnd=fftshift(0.54+0.46*cos(2*pi*n/64))
    y=y*wnd
    y[0]=0
    y=fftshift(y)
    chirped_[:,i]=fftshift(fft(y))/64

x=linspace(-pi,pi,17);x=x[:-1]
y=linspace(-512,512,65);y=y[:-1]

Y,X=meshgrid(x,y)
#Surface Plot
fig = figure(figsize = (10,10))
ax = fig.add_subplot(211, projection = "3d")
surf = ax.plot_surface(X, Y, abs(chirped_), cmap = cm.plasma)
fig.colorbar(surf, shrink = 0.5)
ax.set_title("Surface plot of Magnitude Response vs Frequency and Time")
ax.set_xlabel("Frequency") 
ax.set_ylabel("Time")
ax.set_zlabel(r"Phase of $Y$")

ax = fig.add_subplot(212, projection = '3d')
surf = ax.plot_surface(X, Y, abs(angle(chirped_)), cmap = cm.plasma)
fig.colorbar(surf, shrink = 0.5)
ax.set_title("Surface plot of Phase Response vs Frequency and Time")
ax.set_xlabel("Frequency") 
ax.set_ylabel("Time")
ax.set_zlabel(r"Phase of $Y$")
show()

