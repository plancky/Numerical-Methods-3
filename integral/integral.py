import numpy as np 
from scipy.integrate import quad

def trapezoidal_rule(a,b,f,n=2**4,n_max=2**32,rtol=0.5e-6):
    nstart,nstop = np.log2(n),np.log2(n_max)
    n_array = np.logspace(nstart,nstop,base=2,num = int(nstop-nstart+1))
    I = np.zeros(n_array.shape)
    h = (b-a)/n_array
    x = np.linspace(a,b,int(n_array[0]+1))
    print(x)
    y = f(x)
    I[0] = (h[0]/2)*np.sum(y[:-1] + y[1:])
    for i in np.arange(1,n_array.shape[0]):
        x = np.linspace(a,b,int(n_array[i-1]+1))
        modedx = (x[:-1] +x[1:])/2
        I[i] = (1/2)*I[i-1] + h[i]*(np.sum(f(modedx)))
        if np.abs((I[i]-I[i-1])/I[i]) <= rtol:
            return I[i],0
    return I[-1],-1

def simpsons_rule(a,b,f,n=2**4,n_max=2**32,rtol=0.5e-6):
    nstart,nstop = np.log2(n),np.log2(n_max)
    n_array = np.logspace(nstart,nstop,base=2,num = int(nstop-nstart+1))
    I = np.zeros(n_array.shape)
    h = (b-a)/n_array
    x = np.linspace(a,b,int(n_array[0]+1))
    y = f(x)
    I[0] = (h[0]/3)*np.sum(y[:-1:2] +4*y[1:-1:2] +y[1::2])
    omidy = y[1::2]
    for i in np.arange(1,n_array.shape[0]):
        x = np.linspace(a,b,int(n_array[i]+1))
        midx = x[1::2] 
        midy = f(midx)
        I[i] = I[i-1]/2 + h[i]*(4*np.sum(midy) - 2*np.sum(omidy))/3 
        if np.abs((I[i]-I[i-1])/I[i]) <= rtol:
            return I[i],0
        omidy = midy.copy()
    return I[-1],-1

def simpson_rule(a,b,f,n=2**4,n_max=2**20,rtol=0.5e-6):
    nstart,nstop = np.log2(n),np.log2(n_max)
    n_array = np.logspace(nstart,nstop,base=2,num = int(nstop-nstart+1))
    I = np.zeros(n_array.shape)
    simp_rule = lambda h,k: (h/3)*np.sum(k[:-1:2] +4*k[1:-1:2] +k[1::2])
    h = (b-a)/n_array
    xy = np.zeros(int(12e6),dtype=[('x',np.float64),('y',np.float64)])
    x = np.linspace(a,b,int(n_array[0]+1))
    xy['x'][:len(x)],xy['y'][:len(x)] = x.copy(),f(x.copy())
    I[0] = simp_rule(h[0],xy['y'][:len(x)])
    for i in np.arange(1,n_array.shape[0]):
        prev_n = int(n_array[i-1]+1)
        new_n = int(n_array[i]+1)
        x = xy['x'][:prev_n]
        modedx = (x[:-1]+x[1:])/2
        xy[prev_n:prev_n+len(modedx)] = np.array(list(zip(modedx,f(modedx))),dtype=xy.dtype)
        xy[:new_n] = np.sort(xy[:new_n],order='x')
        y = xy['y'][:new_n]
        I[i] = simp_rule(h[i],y)
        if np.abs((I[i]-I[i-1])/I[i]) <= rtol:
            return I[i],i
    return I[-1],-1

if __name__=="__main__":
    print(simpsons_rule(1,8,lambda x: x**2),quad(lambda x: x**2,1,8))