# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:40:50 2022

@author: harsh
"""

from matplotlib import use
from Myintegration import *
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use("bmh")
use("WebAgg")

# define function
def MyLeguQuad(func,n):
    x,w = roots_laguerre(n)
    return np.dot(w,func(x)*np.exp(x))

MyLeguQuad = np.vectorize(MyLeguQuad)

#
# Validation
#
'''
listp = ['one','two','three']
func = []
P = []
K = []
for i in range(len(listp)):
    k = input(f"function {listp[i]}: ") 
    K.append(k)
    func.append(lambda x,i = i: eval(K[i],{'x':x,'np':np}))

print(func[0](0))

for j in range(len(func)):
    p1 = lambda x,j=j : func[j](x)* np.exp(-x) 
    P.append(p1)
    
mat = np.zeros((2,4))
n_r = [2,4]
mat[:,0] = n_r
for i in range(1,4): 
   mat[:,i] = MyLeguQuad(P[i-1],n_r)

np.savetxt("validate-lag-1162.out",mat,fmt="%.7g",delimiter=",",header="n,$I_1$,$I_2$")
print(mat)
'''
######

I1 = lambda x : np.exp(-x)/(1+x**2)
I2 = lambda x : 1/(1+x**2)


dat = np.zeros((7,3))
dat[:,0] = 2**np.arange(1,8)
n_arr = dat[:,0]
dat[:,1] = MyLeguQuad(I1,n_arr)
dat[:,2] = MyLeguQuad(I2,n_arr)
print(dat)

np.savetxt("quad-lag-1162.out",dat,fmt="%.7g",delimiter=",",header="n,$I_1$,$I_2$")


#simpson 
def laguerre_tol(func,n_max,rtol):
    n_arr = np.arange(5,n_max,5)
    I = np.zeros(n_arr.shape)
    r_err = np.zeros(len(n_arr)-1)

    for i in range(len(n_arr)):
        I[i] =  MyLeguQuad(func, n_arr[i])
        if i==0:
           continue
        r_err[i-1] = abs((I[i] - I[i-1])/(I[i]))
        if r_err[i-1] <= rtol:
           return r_err[:i],I[:i],n_arr[:i]


def simpsonLagu(f,max_b= int(2e16),rtol = 0.5*10**(-4)):
    b = 2**np.arange(1,np.floor(np.log2(max_b)))
    print(b)
    I = np.zeros(b.shape)
    r_err = np.zeros(len(b)-1)
    for i,bi in enumerate(b):
        I[i] = MySimp(f,0,bi,m = int(1e7),d =5)[0]
        if i == 0:
            continue
        r_err[i-1] = abs((I[i] - I[i-1])/(I[i]))
        
        if r_err[i-1]<=rtol:
            return  r_err[:i],I[:i],b[:i]
    return I,b





rl_l, inte_l,n_ar = laguerre_tol(I2,500,0.5*10**(-4))
rl_s, inte_s,b_ar = simpsonLagu(I2,int(2e16),0.5*10**(-4))

slope, intercept, r_value, p_value, std_err = stats.linregress(rl_s[3:],inte_s[3:])
Y = slope*rl_s + intercept

print('Slope of relative tol vs Integral line for function 1/(1+x^2)',slope)

#Plotting and comparison 

fig,ax = plt.subplots()
ax.plot(rl_l,inte_l,'--o',label ='improper integral by Laguerre')
ax.plot(rl_s[3:],inte_s[3:],'* ',label = 'improper integral by simpson')
ax.plot(rl_s[3:],Y[3:],label = 'Regression line for simpson')
ax.set_xlabel('relative tolerence')
ax.set_ylabel('Integral')
ax.set_title(r'Comparsion between integral of $f(x)=\frac{1}{1+x^2}$ calculated by simpson and laguerre')
ax.legend()

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(n_ar,inte_l,'--v')
ax1.set_xlabel('n point formula used')
ax1.set_ylabel('Integral')
ax1.set_title('(a)')
ax2.plot(b_ar,inte_s,'r--x')
ax2.set_xlabel('Upper limit on the integral')
ax2.set_ylabel('Integral')
ax2.set_title('(b)')

plt.show()



