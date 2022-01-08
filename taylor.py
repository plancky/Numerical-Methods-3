import matplotlib.pyplot as plt
import math
from numba.np.ufunc import parallel
import numpy as np
import numba as nb
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
'''

#@nb.guvectorize(["void(float64[:],float64,int32,float64[:])"],'(n),(),()->(n)',target="parallel",nopython=True)
@nb.vectorize
def exp(x,a,n):
    sum_ = 1
    for i in np.arange(1,n):
        sum_ += (x-a)**(i)/math.gamma(i+1)
    return(sum_)

@nb.vectorize
def MySinSeries(x,a,n):
    sum_ = 0
    for i in np.arange(n):
        sum_ += (-1)**i*(x-a)**(2*i+1)/math.gamma(2*i+2)
    return(sum_)

@nb.vectorize
def MyCosSeries(x,a,n):
    sum_ = 0
    for i in np.arange(n):
        sum_ += (-1)**i*(x-a)**(2*i)/math.gamma(2*i+1)
    return(sum_)
#MySinSeries = lambda x,a,n : (exp(1j*x,a,n) - exp(-1j*x,a,n))/2j
#MyCosSeries = lambda x,a,n : (exp(1j*x,a,n) + exp(-1j*x,a,n))/2

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
    m = np.arange(2,22,2)
    fig1,(ax1,ax2) = plt.subplots(1, 2)
    yj_sin = np.array([MySinSeries(xs,0,i) for i in m],dtype=float)
    for j in range(len(m)) : 
        ax1.plot(xs,yj_sin[j],label=f"m={m[j]}")
    ax1.plot(xs,np.sin(xs))

    yj_cos = np.array([MyCosSeries(xs,0,i) for i in m],dtype=float)
    for j in range(len(m)) : 
        ax2.plot(xs,yj_cos[j],label=f"m={m[j]}")
    ax2.plot(xs,np.cos(xs))
    fig1.legend()
    plt.plot()
    plt.show()

    '''
    fig,ax1 = plt.subplots(1, 1)
    rtol = 0.5e-10
    old = exp(xs,0,1)
    setaxis(ax1)
    for i in range(2,50):
        plt.plot(xs,old,label=f"{i}")
        new = exp(xs,0,i)
        relative_error = np.max(np.abs((new - old)/new))
        if relative_error <= rtol :
            print(f"found at {i} {relative_error}")
            break
        old = new.copy()
    plt.legend()
    plt.show()'''