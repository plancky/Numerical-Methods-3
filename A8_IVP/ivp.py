import numpy as np
from Myintegration import *
import matplotlib.pyplot as plt
from matplotlib import use
#from numba import njit,vectorize
use("webAgg")
plt.style.use("bmh")

def put(x,i,v):
    x_ = x.copy()
    np.put(x_,i,v)
    return(x_)

def rk4(y,f,h,i):
    x_i,t = y[i],y[0] 
    m1 = f[i-1](*y)
    m2 = f[i-1](*put(y,[0,i],[t+h/2,x_i+h/2*m1]))
    m3 = f[i-1](*put(y,[0,i],[t+h/2,x_i+h/2*m2]))
    m4 = f[i-1](*put(y,[0,i],[t+h,x_i+h*m3]))
    avg_m = ( m1 + 2*(m2+m3) + m4 )/6
    return(x_i + h*avg_m)

def rk2(i,y,f,h):
    return(y[i] + (f[i-1](*y) + f[i-1](*put(y,[0,i],[y[0]+h,y[i] + h*f[i-1](*y)])
    ))/2*h)

def em(i,y,f,h):
    return(y[i] + f[i-1](*put(y,[0,i],[y[0]+h/2,y[i] + (h/2)*f[i-1](*y)]))*h)

def ef(i,y,f,h):
    return(y[i] + h* f[i-1](*y))

def ode_solver(func,t_axis,ini,Nh,n,method=rk4):
    N,h = Nh
    data = np.zeros((N+1,n),dtype="double")
    data[:,0],data[0,:] = t_axis, np.array(ini)
    params =  np.arange(1,n)
    iter = np.vectorize(method, excluded=["f","h","y"]) 
    for j in range(1,N+1) :
        data[j,1:] = iter(y=data[j-1],f=func,h=h,i=params)
    return(data)

class set_problem:
    def __init__(self,f,dom,ini,N,vars=None):
        self.n = len(ini) 
        if self.n-1 != len(f) :
            raise ValueError("unequal equations and parameters") 
        self.dom = np.linspace(*dom,N+1)
        self.ini = ini
        self.f = f
        if vars is not None :
            self.vars = vars
        else:
            self.vars = list(range(self.n))
        self.ivp = [self.f,self.dom,self.ini,(N,self.dom[1] - self.dom[0]),self.n]
        self.dat = dict()
    def pass_analytic(self,x):
        self.analytic = x

    def comp_E(self):
        self.E_dat = dict()
        analytic_dat = np.zeros((len(self.dom),self.n-1))
        for i,j in enumerate(self.analytic):
            analytic_dat[:,i] = j(self.dom)

        for i in self.dat:
            self.E_dat[i] = np.max(np.abs(analytic_dat - self.dat[i][:,1:]),axis=0)
        return(self.E_dat) 

    def rk4(self):
        self.data_rk4 = ode_solver(*self.ivp,method=rk4)
        self.dat["rk4"] = self.data_rk4
        return(self.data_rk4)
    def rk2(self):
        self.data_rk2 = ode_solver(*self.ivp,method=rk2)
        self.dat["rk2"] = self.data_rk2
        return(self.data_rk2)
    def em(self):
        self.data_em = ode_solver(*self.ivp,method=em)
        self.dat["em"] = self.data_em
        return(self.data_em)
    def ef(self):
        self.data_ef = ode_solver(*self.ivp,method=ef)
        self.dat["ef"] = self.data_ef
        return(self.data_ef)
    def jt_plot(self,ax,j,ladd=""):
        for i in self.dat:
            ax.plot(self.dom,self.dat[i][:,j],"-1",label=f"{i}--{self.vars[j]}"+ladd)
    def kj_plot(self,ax,j,k):
        for i in self.dat:
            ax.plot(self.dat[i][:,j],self.dat[i][:,k],"-1",label=f"{i}--{self.vars[k]} vs {self.vars[j]}")

if __name__ == "__main__" : 
    import matplotlib.pyplot as plt
    plt.style.use("bmh")
    ivp3_args = [[lambda x,y1,y2,y3 : y2 - y3 +x , lambda x,y1,y2,y3 : 3*x**2, lambda x,y1,y2,y3 : y2 + np.exp(-x) ],
                    [0,1],
                    (0,1,1,-1),500]
    
    y1 = lambda x :  -0.05*x**5 + 0.25*x**4 + x + 2 - np.exp(-x)
    y2 = lambda x :  x**3 +1
    y3 = lambda x :  0.25*x**4 +x - np.exp(-x)

    ys = [y1,y2,y3]
    N = 10**np.arange(5)
    xf = [1,2.5,5,7.5,10]
    E_dat = np.ones((len(N),3,5))
    for i,Ni in enumerate(N):
        ivp3_args[-1] = Ni
        for j,xfj in enumerate(xf):
            ivp3_args[1][1] = xfj
            ivp3 = set_problem(*ivp3_args)
            
            ivp3.ef()
            ivp3.pass_analytic(ys)  
            E_dat[i,:,j] = ivp3.comp_E()["ef"]

    lls= {0:"$y_1$",1:"$y_2$",2:"$y_3$"}
    fig2,a = plt.subplots(1,1)
    x_space = np.linspace(0,10,200)
    for i,yi in enumerate(ys):
        ivp3.jt_plot(a,i+1)
        a.plot(x_space,yi(x_space),label=lls[i])
    
    a.legend()
    fig2,a = plt.subplots(1,3)
    for j in np.arange(E_dat.shape[2]):
        E = E_dat[:,:,j] 
        for i in np.arange(E.shape[1]):
            logE = np.log10(E_dat[:,i])
            logE[logE==-1*np.inf]=0 
            a[i].plot(np.log10(N),np.log10(E[:,i]),"-1",label=str(xf[j]))
    
    for i,ax in enumerate(a):
        ax.set_xlabel("$log_{10}N$");ax.set_ylabel("$log_{10}E$");
        ax.set_title(lls[i])
        ax.legend()

    ivp4_args = [[lambda x,y1,y2: y2,lambda x,y1,y2: 2*y2 - 2*y1 + np.exp(2*x)*np.sin(x)],
                [0,1],
                (0,-0.4,-0.6),5]
    ivp4 = set_problem(*ivp4_args)
    print(ivp4.rk2())

    fig2,a = plt.subplots(1,1)
    ivp4.jt_plot(a,1)
    yt = lambda x : (np.exp(2*x)*(np.sin(x) - 2*np.cos(x)))/5
    xs = np.linspace(0,1,6)
    print(yt(xs))
    plt.show()
    
    '''
    ivp.rk4()
    ivp2.rk4()
    ivp1.rk4() 
    #fig,(ax1,ax2) = plt.subplots(2,1)
    fig2,a = plt.subplots(1,1)
    ivp.kj_plot(a,1,2)
    ivp1.kj_plot(a,1,2)
    ivp2.kj_plot(a,1,2)
    #ax1.legend(),ax2.legend(),
    a.legend()
    plt.show()'''