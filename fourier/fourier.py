from scipy.fft import fft  
from scipy.integrate import quad 
import numpy as np
from svgpathtools import svg2paths
import matplotlib.pyplot as plt

PATH = "/home/planck/Pictures/me/ankur.svg" 
JSPATH = "/home/planck/Desktop/mp-3/fourier/dat.js"
NAME = "ankur"
class curve:
    def __init__(self,path,name) -> None:
        p,a = svg2paths(path)
        self.name = name
        self.funcs=[]
        for single_curve in p:
            for j in single_curve:
                self.funcs.append(j.poly())        
        pass

    def f(self,t):
        n = len(self.funcs)
        for i in range(1,n+1):
            if t<= i/n:
                return(self.funcs[i-1](n*t-i+1))

    def write_coeff(self,jspath,dat):
        with open(jspath,"a") as f:
            f.write(f"\nconst {me.name} = "+str(dat)+";\n")
            f.write("export {"+f"{me.name}"+"} ;")

def simpson(a, b, y=None, N=10, f=lambda x: np.sin(x)):
    integral = 0
    x = np.linspace(a, b, N + 1)
    if y is None:
        y = f(x)
    y = np.array(y)
    l = len(y)
    if len(x) != l:
        raise ValueError("length of x must be same as y")

    y_1=y[:l-2:2]
    y_2=y[1:l-1:2]
    y_3=y[2:l:2]
    integral = (b-a)/(N*3) * (np.sum(y_1 + 4*y_2 + y_3))
    return integral

def fourier_coeff(n,g,a=0,b=1):
    r = quad(lambda t: np.real(g(t)*np.exp((-2j*np.pi*n*t))),a,b)
    i = quad(lambda t: np.imag(g(t)*np.exp((-2j*np.pi*n*t))),a,b)
    return(r[0]+1j*i[0])
def fourier_coeff2(n,g):
    L= len(g)
    simp = np.vectorize(lambda f: simpson(0,1,None,10,lambda t: np.exp(-2j*np.pi*n*t/L)*f(t))) 
    return(np.sum(simp(g))/L)

def fourier_coeff3(n,g,a=0,b=1):
    r = simpson(a,b,None,1500,lambda t: g(t)*np.exp((-2j*np.pi*n*t)))
    return(r)

fourier_coeff  = np.vectorize(fourier_coeff)
fourier_coeff3  = np.vectorize(fourier_coeff3)

me = curve(PATH,NAME)
param_c = np.vectorize(me.f)

freq = np.arange(-70,70)
coef = fourier_coeff3(freq,param_c,0,1)
mag = np.abs(coef)
dat = np.sort(np.array(list(zip(np.real(coef),np.imag(coef),mag,freq)),dtype = [('x',np.float64),('y',np.float64),('m',np.float64),('f',np.int64)]),order = 'm')[::-1]
dat = list(np.array(dat,dtype = np.ndarray))
for i,d in enumerate(dat):
    dat[i]= list(dat[i])

me.write_coeff(JSPATH,dat)

##print(np.sort(coef))
#t_space = np.linspace(0,1,int(2e3))
#c = param_c(t_space)
#plt.plot(np.real(c),np.imag(c))
#plt.show()