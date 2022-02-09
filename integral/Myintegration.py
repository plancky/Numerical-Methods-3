import numpy as np 

def MyTrap(func,a,b,n,d=None):
    if d is not None:
        max_n = np.floor(np.log2(n))
        n_array = np.logspace(1,max_n,base=2,num = int(max_n))
        I = np.zeros(n_array.shape)
        h = (b-a)/n_array
        x = np.linspace(a,b,int(n_array[0]+1))
        y = func(x)
        I[0] = (h[0]/2)*np.sum(y[:-1] + y[1:])
        for i in np.arange(1,n_array.shape[0]):
            x = np.linspace(a,b,int(n_array[i]+1))
            midx  = x[1::2]
            I[i] = (1/2)*I[i-1] + h[i]*(np.sum(func(midx)))
            if np.abs(I[i]-I[i-1]) <= 0.5/10**d*np.abs(I[i]):
                return I[i],n_array[i]
        print("Could not reach desired accuracy with the given upperlimit on the number of intervals. ")
        return I[-1],n_array[-1]
    x = np.linspace(a,b,n+1)
    y = func(x)
    return((b-a)/(2*n)*np.sum(y[:-1]+y[1:]))

def MySimp(func,a,b,n,d=None):
    if d is not None:
        max_n = np.floor(np.log2(n))
        n_array = np.logspace(1,max_n,base=2,num = int(max_n))
        I = np.zeros(n_array.shape)
        h = (b-a)/n_array
        x = np.linspace(a,b,int(n_array[0]+1))
        y = func(x)
        I[0] = (h[0]/3)*np.sum(y[:-1:2] +4*y[1:-1:2] +y[2::2])
        omidy = y[1::2]
        for i in np.arange(1,n_array.shape[0]):
            x = np.linspace(a,b,int(n_array[i]+1))
            midx = x[1::2] 
            midy = func(midx)
            I[i] = I[i-1]/2 + h[i]*(4*np.sum(midy) - 2*np.sum(omidy))/3 
            if np.abs(I[i]-I[i-1]) <= 0.5/10**d*np.abs(I[i]):
                return I[i],n_array[i]
            omidy = midy.copy()
        print("Could not reach desired accuracy with the given upperlimit on the number of intervals. ")
        return I[-1],n_array[-1]
    x = np.linspace(a,b,n+1)
    y = func(x)
    return((b-a)/(3*n)*np.sum(y[:-1:2]+4*y[1::2]+y[2::2]))

MyTrap = np.vectorize(MyTrap)
MySimp = np.vectorize(MySimp)


#def Gaussquadrature():
    