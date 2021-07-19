"""
        EE2703 Applied Programming Lab - 2021
            Assignment 4 solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE2703_ASSIGN4_EE19B124
            commandline INPUT : <FILE NAME>
"""
import math
import numpy as np
from pylab import *
import scipy.integrate as integral
import scipy
#Q1
# Required Functions
def f(x):
    return np.exp(x)
def g(x):
    return np.cos(np.cos(x))
def u(x,k,f):
    return f(x)*np.cos(k*x)
def v(x,k,f):
    return f(x)*np.sin(k*x)
#Original and expected functions from fourier series
x = np.arange(-2*np.pi,4*np.pi,0.01)
a = np.arange(-2*np.pi,4*np.pi,0.01)
for i in range(len(a)):
    if a[i] > 2*np.pi or a[i] < 0:
        a[i] = a[i] - (2*pi)*(a[i]//(2*np.pi))
figure(1,figsize=(10,10))    #plot of exp(x) in semilogy axis
semilogy(x,f(x),label = 'Actual Graph')

figure(2,figsize=(10,10)) #plot of cos(cos(x))
plot(x,g(x),label = 'Actual Graph')

figure(1)
semilogy(x,f(a),label = 'Graph obtained by fourier Analysis')#expected plot from fourier series
title("Fig 1:Actaul vs expected on semilog scale of exp(x)")
xlabel("x",fontsize = 15)
ylabel("y(logscale)",fontsize = 15)
legend()
grid()

figure(2)
plot(x,g(a),label = 'Graph obtained by fourier Analysis') #expected plot from fourier series
legend(loc = 'upper right')
title("Fig 2:Actaul vs expected of cos(cos(x))")
xlabel("x",fontsize = 15)
ylabel("y",fontsize = 15)
grid()

#Q2 Computing the Coefficients

bf_0 = 0
af_0 = integral.quad(f,0,2*np.pi)[0] / (2*np.pi)
coefficients_f = [af_0,bf_0] #first 51 coefficients of exp(x)
for i in range(1,26):
    a_n = integral.quad(u,0,2*pi,args = (i,f))[0] / np.pi
    b_n = integral.quad(v,0,2*pi,args = (i,f))[0] / np.pi
    temp = [a_n,b_n]
    coefficients_f = coefficients_f + temp
bg_0 = 0
ag_0 = integral.quad(g,0,2*np.pi)[0] / (2*np.pi)
coefficients_g = [ag_0,bg_0]#first 51 coefficients of cos(cos(x))
for i in range(1,26):
    a_n = integral.quad(u,0,2*pi,args = (i,g))[0] / np.pi
    b_n = integral.quad(v,0,2*pi,args = (i,g))[0] / np.pi
    temp = [a_n,b_n]
    coefficients_g = coefficients_g + temp
af_n = []
bf_n = []
for i in range(52):
    if i%2 == 0:
        af_n = af_n +[coefficients_f[i]]
    else:
        bf_n = bf_n + [coefficients_f[i]]
ag_n = []
bg_n = []
for i in range(52):
    if i%2 == 0:
        ag_n = ag_n +[coefficients_g[i]]
    else:
        bg_n = bg_n + [coefficients_g[i]]
coefficients_f.remove(coefficients_f[1])
coefficients_g.remove(coefficients_g[1])

#Q3 Plotting magnitude of coefficients vs n

figure(3,figsize=(10,10))       #plot of coefficients  of exp(x) in semilog axis
semilogy(range(26),af_n,'ro',label='a_n of exp(x)',markersize=4)
semilogy(range(1,26),np.abs(bf_n[1:]),'bo',label='b_n of exp(x)',markersize=6)
figure(4,figsize=(10,10))   #plot of coefficients  of exp(x) in loglog axis
loglog(range(26),af_n,'ro',label='a_n of exp(x)',markersize=4)
loglog(range(1,26),np.abs(bf_n[1:]),'bo',label='b_n of exp(x)',markersize=6)
figure(5,figsize=(10,10))          #plot of coefficients  of cos(cos(x)) in semilog axis
semilogy(range(26),np.abs(ag_n),'ro',label='a_n of cos(cos(x))',markersize=4)
semilogy(range(1,26),np.abs(bg_n[1:]),'bo',label='b_n of cos(cos(x))',markersize=4)
figure(6,figsize=(10,10))           #plot of coefficients  of cos(cos(x)) in loglog axis
loglog(range(26),np.abs(ag_n),'ro',label='a_n of cos(cos(x))',markersize=4)
loglog(range(1,26),np.abs(bg_n[1:]),'bo',label='b_n of cos(cos(x))',markersize=4)

#Q4 Computing the coefficients using the least squares approach

x2=linspace(0,2*pi,401)
x2=x2[:-1] # drop last term to have a proper periodic integral
b_f=f(x2)# f has been written to take a vector
b_g = g(x2)
A=zeros((400,51)) # allocate space for A
A[:,0]=1 # col 1 is all ones
for k in range(1,26):
    A[:,2*k-1]=cos(k*x2) # cos(kx) column
    A[:,2*k]=sin(k*x2) # sin(kx) column
#endfor
c1=lstsq(A,b_f,rcond = None)[0]# the ’[0]’ is to pull out the best fit vector. lstsq returns a list.
c2 = lstsq(A,b_g,rcond = None)[0]
pred_af = [c1[0]] #predicted coefficients
pred_bf = []
for i in range(1,51):
    if i%2 == 0:
        pred_bf = pred_bf + [c1[i]]
    else:
        pred_af = pred_af + [c1[i]]
pred_ag = [c2[0]]  #predicted coefficients
pred_bg = []
for i in range(1,51):
    if i%2 == 0:
        pred_bg = pred_bg + [c2[i]]
    else:
        pred_ag = pred_ag + [c2[i]]

#Q5 PLotting the coefficients of f(x),g(x) in semilog,loglog axis

figure(3)
semilogy(range(26),np.abs(pred_af),'go',label='predicted_an of exp(x)',markersize=4)
semilogy(range(1,26),np.abs(pred_bf),'yo',label='predicted_bn of exp(x)',markersize=4)
legend()
title("Fig 3:Calculated vs predicted coeff of exp(x)")
xlabel("n",fontsize = 15)
ylabel("Coefficients(semilog)",fontsize = 15)
grid()

figure(5)
semilogy(range(26),np.abs(pred_ag),'go',label='predicted_an of cos(cos(x))',markersize=4)
semilogy(range(1,26),np.abs(pred_bg),'yo',label='predicted_bn of cos(cos(x))',markersize=4)
legend()
title("Fig 5:Calculated vs predicted coeff of cos(cos(x))")
xlabel("n",fontsize = 15)
ylabel("Coefficients(semilolg)",fontsize = 15)
grid()

figure(4)
loglog(range(26),np.abs(pred_af),'go',label='predicted_an of exp(x)',markersize=4)
loglog(range(1,26),np.abs(pred_bf),'yo',label='predicted_bn of exp(x)',markersize=4)
legend()
title("Fig 4:Calculated vs predicted coeff of exp(x)")
xlabel("n",fontsize = 15)
ylabel("Coefficients(loglog)",fontsize = 15)
grid()

figure(6)
loglog(range(26),np.abs(pred_ag),'go',label='predicted_an of cos(cos(x))',markersize=4)
loglog(range(1,26),np.abs(pred_bg),'yo',label='predicted_bn of cos(cos(x))',markersize=4)
legend()
title("Fig 6:Calculated vs predicted coeff of cos(cos(x))")
xlabel("n",fontsize = 15)
ylabel("Coefficients(loglog)",fontsize = 15)
grid()

#Q6

deviation_f = max(np.abs(coefficients_f - c1))
deviation_g = max(np.abs(coefficients_g - c2))
print("The Coefficients obtained from both the methods are Different from the Graph")
print("The maximum deviation betwwen the coefficients of exp(x) is {}".format(deviation_f))
print("The maximum deviation betwwen the coefficients of cos(cos(x)) is {}".format(deviation_g))

#Q7

figure(7,figsize=(10,10))
semilogy(x2,f(x2),'bo',label='exp(x)',markersize=2)
semilogy(x2,(np.array(A) @ np.array(coefficients_f)),'ro',markersize=4,label = 'Fourier')
semilogy(x2,(np.array(A) @ np.array(c1)),'go',markersize=4,label = 'lstsq')
legend()
title("Calculated vs predicted functions of exp(x)")
xlabel("x",fontsize = 15)
ylabel("y(semilog)",fontsize = 15)
grid()

figure(8,figsize=(10,10))
semilogy(x2,g(x2),'yo',label='cos(cos(x))',markersize=6)
semilogy(x2,(np.array(A) @ np.array(coefficients_g)),'ro',markersize=5,label = 'Fourier')
semilogy(x2,(np.array(A) @ np.array(c2)),'go',markersize=3.5,label = 'lstsq')
legend()
title("Calculated vs predicted fittings of cos(cos(x))")
xlabel("x",fontsize = 15)
ylabel("y(semilog)",fontsize = 15)
grid()