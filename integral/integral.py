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

if __name__=="__main__":
    print(trapezoidal_rule(1,2,lambda x: x**2),quad(lambda x: x**2,1,2))