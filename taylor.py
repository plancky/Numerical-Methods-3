import matplotlib.pyplot as plt
import math
import numpy as np
from numba import vectorize
import pandas as pd
plt.style.use("seaborn-dark-palette")

'''
def exp(x,a,n):
    f = np.vectorize(lambda x,a,i : ((x-a)**i)/np.math.factorial(i),excluded =[0,1])
    return(np.sum(f(x,a,np.arange(n))))
#exp = lambda x,a,n : np.sum([((x-a)**i)/np.math.factorial(i) for i in np.arange(n)]) 
#exp = np.vectorize(exp,excluded=[1,2])
'''
'''
def MySinSeries(x,a,n):
    f = np.vectorize(lambda x,a,i : ((-1)**i*(x-a)**(2*i+1))/np.math.factorial(2*i+1),excluded =[0,1])
    return(np.sum(f(x,a,np.arange(n))))
MySinSeries = np.vectorize(MySinSeries,excluded=[1,2])
#MySinSeries = lambda x,a,n : (exp(1j*x,a,n) - exp(-1j*x,a,n))/2j
#MyCosSeries = lambda x,a,n : (exp(1j*x,a,n) + exp(-1j*x,a,n))/2


#@nb.guvectorize(["void(float64[:],float64,int32,float64[:])"],'(n),(),()->(n)',target="parallel",nopython=True)
'''

@vectorize
def exp(x,a,n):
    sum_ = 1
    for i in np.arange(1,n):
        sum_ += (x-a)**(i)/math.gamma(i+1)
    return(sum_)

@vectorize
def MySinSeries(x,a,n):
    sum_ = 0
    for i in np.arange(n):
        sum_ += (-1)**i*(x-a)**(2*i+1)/math.gamma(2*i+2)
    return(sum_)

@vectorize
def MyCosSeries(x,a,n):
    sum_ = 0
    for i in np.arange(n):
        sum_ += (-1)**i*(x-a)**(2*i)/math.gamma(2*i+1)
    return(sum_)

def get_n_sin(x,rtol = 0.5e-4):
    i_ = np.zeros(x.shape)
    y1_,max_rel_ = i_.copy(),i_.copy()
    for k in np.arange(len(x)):        
        y0 = MySinSeries(x[k],0,1)
        for i in range(2,1000):
            y1 = MySinSeries(x[k],0,i)
            max_rel = np.max(np.abs((y1 - y0)/y1))
            if max_rel <= rtol :
                i_[k] = i ; y1_[k] = y1 ; max_rel_[k] = max_rel
                break
            elif max_rel == 0:
                i_[k] = 0 ; y1_[k] = y1 ; max_rel_[k] = 0
            y0 = y1.copy()
    return([i_,y1_,max_rel_])

def setaxis(ax, title="",x=["",""]):
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.legend(loc="lower right")
    ax.set_title(title)
    ax.grid()

if __name__ == "__main__":
    xs= np.linspace(-2*np.pi,2*np.pi,1000)
    m = np.arange(2,21,2)
    fig1,(ax1,ax2) = plt.subplots(1, 2)
    yj_sin = np.array([MySinSeries(xs,0,i) for i in m],dtype=float)
    
    for j in range(len(m)) : 
        ax1.plot(xs,yj_sin[j],label=f"m={m[j]}")
    ax1.plot(xs,np.sin(xs),label="Numpy's sin(x) ")
    ax1.set_ylim([-10, 10])
    
    setaxis(ax1,"$\sin(x)$",["x","y"])
    ym_sin = MySinSeries(np.pi/4,0,m) 
    ax2.plot(m,ym_sin,"-*",label=r"MySinSeries($\frac{\pi}{4},m$)")
    ax2.plot(m,np.sin(np.pi/4)*np.ones(m.shape),"-",label=r"Numpy's $\sin(\frac{\pi}{4})$")
    ax2.legend()

    ax1.set_xlabel("x");ax1.set_ylabel("y")
    ax2.set_xlabel("m");ax2.set_ylabel(r"$\sin(\frac{\pi}{4})$")

    fig2,(ax12,ax22) = plt.subplots(1, 2)

    yj_cos = np.array([MyCosSeries(xs,0,i) for i in m],dtype=float)
    for j in range(len(m)) : 
        ax12.plot(xs,yj_cos[j],label=f"m={m[j]}")
    ax12.plot(xs,np.cos(xs),label="Numpy's cos(x) ")
    ax12.set_ylim([-10, 10])
    
    setaxis(ax12,"$\cos(x)$")
    
    ax12.set_xlabel("x");ax12.set_ylabel("y")

    ym_cos= MyCosSeries(np.pi/4,0,m) 
    ax22.plot(m,ym_cos,"-*",label=r"MyCosSeries($\frac{\pi}{4},m$)")
    ax22.plot(m,np.cos(np.pi/4)*np.ones(m.shape),"-",label=r"Numpy's $\cos(\frac{\pi}{4})$")
    ax22.set_xlabel("m");ax22.set_ylabel(r"$\cos(\frac{\pi}{4})$")
    ax22.legend()

    
    xvec = np.arange(0,np.pi+0.1,np.pi/8)
    reltol = 0.5e-6
    n,calsin,relerror =get_n_sin(xvec,reltol)

    table = pd.DataFrame({"x": xvec , "MySinSeries(x)" :map(lambda x: f"{x:#.9g}",calsin),"n":n ,"Numpy's sin(x)":map(lambda x: f"{x:#.9g}",np.sin(xvec))})
    table.to_csv("table.csv")
    
    fig0,ax0 = plt.subplots(1, 1)
    xs2 = np.linspace(0,np.pi)
    ax0.plot(xs2,np.sin(xs2),label= "Numpy's sin(x) continuous")
    ax0.scatter(xvec,list(map(lambda x: float(f"{x:#.3g}"),calsin)),label = "MySinSeries() with 3 significant digits")
    ax0.set_xlabel("x");ax0.set_ylabel("y");ax0.grid()
    print(table)

    plt.plot()
    plt.show()

    '''
    fig,ax1 = plt.subplots(1, 1)
    plt.legend()
    plt.show()'''
