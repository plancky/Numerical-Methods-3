from Myintegration import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use("webAgg")
plt.style.use("bmh")

def put(x,i,v):
    x_ = x.copy()
    np.put(x_,i,v)
    return(x_)

def rk4(f,t,y,h): 
    m1 = f(t,y)
    m2 = f(t+h/2,y+h/2*m1)
    m3 = f(t+h/2,y+h/2*m2)
    m4 = f(t+h,y+h*m3)
    avg_m = ( m1 + 2*(m2+m3) + m4 )/6
    return(y + h*avg_m)

def rk2(f,t,y,h):
    return(y+(f(t,y) + f(t+h,y + h*f(t,y))
    )/2*h)

def em(f,t,y,h):
    return(y + h*f(t+h/2,y+(h/2)*f(t,y)))

def ef(f,t,y,h):
    return(y + h*f(t,y))

def ode_solver(func,t_axis,ini,Nh,n,method=ef):
    N,h = Nh
    data = np.zeros((N+1,n),dtype="double")
    data[:,0],data[0,:] = t_axis, np.array(ini)
    params =  np.arange(1,n)
    iter = np.vectorize(method, excluded=["f","h","y"]) 
    for j in range(1,N+1) :
        data[j,1:] = iter(y=data[j-1,1:],t=data[j-1,0],f=func,h=h)
    return(data)

class set_problem:
    def __init__(self,f,dom,ini,N,vars=None):
        self.n = len(ini) 
        #if self.n-1 != len(f) :
        #    raise ValueError("unequal equations and parameters") 
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


if __name__ == "__main__":
    from matplotlib import use
    use("WebAgg")
    import matplotlib.pyplot as plt
    func = lambda t,y: np.array([1]).dot(y)
    p = set_problem(func,[0,6],[0,1],100)
    p.rk4();p.rk2();p.em();p.ef()
    fig,ax = plt.subplots(1,1)
    p.jt_plot(ax,1)
    plt.show()

