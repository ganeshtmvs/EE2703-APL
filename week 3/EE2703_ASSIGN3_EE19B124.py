"""
        EE2703 Applied Programming Lab - 2019
            Assignment 3 solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE2703_ASSIGN3_EE19B124
            commandline INPUT : <FILE NAME>
"""
# script to generate data files for the least squares assignment
import numpy
from numpy import *
from pylab import *
import scipy
import scipy.special as sp

#Question1

N = 101                                   # no of data points
k = 9                                     # no of sets of data with varying noise

# generate the data points and add noise

t = linspace(0,10,N)                      # t vector
y = 1.05*sp.jn(2,t)-0.105*t               # f(t) vector[True value without noise]
Y = meshgrid(y,ones(k),indexing='ij')[0]  # make k copies
scl = logspace(-1,-3,k)                     # noise stdev
n = dot(randn(N,k),diag(scl))               # generate k vectors
yy = Y+n                                     #Data with noise added
savetxt("fitting.dat",c_[t,yy])         # write out matrix to file

#Question2

DATA = loadtxt('fitting.dat',delimiter =' ') #EXTRACTING THE DATA
TIME = DATA[:,0]

#Question 3,4

def g(time,A,B):                            #function to calculate the expression without noise
    result = A*(sp.jn(2,time))+B*(time)
    return result

TRUE_VALUE = g(TIME,1.05,-0.105)
sub_s = ['\u2080','\u2081','\u2082','\u2083','\u2084','\u2085','\u2086','\u2087','\u2088','\u2089'] #unicode of subscripts
episilon_uni  ='\u03B5'
sigma_uni = '\u03C3'
# shadow plot
figure(figsize= (10,10))
xlabel(r'$t$',size=20)
ylabel(r'$f(t)+n$',size=20)
title(r'Q4:Plot of the data to be fitted',size=20)
for i in range(1,10):          # Graph of data with different noise
    plot(TIME,DATA[:,i],label = 'σ'+sub_s[i]+str('=') + str(round(scl[i-1],4)),alpha = 0.8)
plot(TIME,TRUE_VALUE,label = 'True Value',color = "b")
legend(loc = 'lower left',ncol=1,title='Sigma values')
grid(True)
show()

#Question5

figure(figsize=(10,10))  #Graph of error of  1st column of data with that of True value
xlabel(r'$t$',size=20)
ylabel(r'$Data$',size=20)
title(r'Q5:Data Points for σ along with exact function',size=20)
for i in range(9):
    errorbar(TIME[::5],DATA[:,i+1][::5],scl[i],fmt='o',label='for'+ 'σ'+sub_s[i+1])
plot(TIME,TRUE_VALUE,label = 'f(t)[True Value]')
grid(True)
legend(loc = 'upper right',ncol= 2,fontsize = 10,title = 'Error bars')
show()

#Question6

J_vector = []
for i in range(0,N):
    J_vector.append(sp.jn(2,TIME[i]))
M = c_[J_vector,TIME]
p = c_[[1.05,-0.105]]
Matrix_value = np.array(np.dot(M,p)) #Calculating data from dot product of vectors
formula_value =c_[g(TIME,1.05,-0.105)] #Calculating data from the formula
if np.array_equal(Matrix_value,formula_value):
    print("They are Equal")
else:
    print('Check if you have changed something above')

#Question7

Error_matrix = [np.zeros(21) for i in range(21)] #errors with all different values of A and B (21*21 values)
A = linspace(0,2,21)                              #with respect to 1st column of Data
B = linspace(-0.2,0,21)
for i in range(21):
    for j in range(21):
        Error_matrix[i][j] = np.sum((DATA[:,1] - g(TIME,A[i],B[j]))**2)*(1/101)

#Question8

figure(figsize=(10,10))
val = contour(A,B,Error_matrix,15)  #Contour plot of error (20 lines will be plotted)
plot(1.05,-0.105,marker = 'o',color = 'b')
text(1.07,-0.105,'True Value',color = 'g',size=15)
clabel(val,val.levels[:6],fontsize=10)
xlabel('A',size=20)
ylabel('B',size=20)
title('Q8:Contour plot of εᵢⱼ',size=20)

#Question 9

A_err = [np.zeros(1) for i in range(9)]
B_err = [np.zeros(1) for i in range(9)]
for i in range(1,10):
    p=lstsq(M,DATA[:,i],rcond=None)  #estimating the A,B which fits for data with different noises
    A_err[i-1]=abs(p[0][0]-1.05)      #error of A,B with that of A,B corresponding to true value
    B_err[i-1]=abs(p[0][1]+0.105)

#Question 10

figure(figsize=(10,10))
for i in range(9):
    axvline(scl[i],color='g',alpha = 0.5) # plotting the error values with that of sigma values
xlabel('Standard Deviation of Noise',size=20)
ylabel('MS error',size=20)
title('Variation of error with noise',size = 20)
plot(scl,A_err,'--ro',label='Aerror')
plot(scl,B_err,'--bo',label='Berror')
legend(loc = 'upper left',fontsize = 15)
grid(True)
show()

#Question 11

figure(figsize =(10,10))   #plotting the above graph in log scale
xlabel('Standard Deviation of Noise',size=20)
ylabel('MS error',size=20)
title('Variation of error with noise in log scale',size = 20)
for i in range(9):
    axvline(scl[i],color='g',alpha = 0.5)
loglog(scl,A_err,'--ro',label='Aerror')
loglog(scl,B_err,'--bo',label='Berror')
legend(loc = 'upper left',fontsize = 15)
show()