from numpy import linspace
from Myintegration import *


def FourierCoeff(func,n,L,d,method="simp",even_flag=-1):
    methods = {"trap":MyTrap,"simp":MySimp,"quad":MyLegQuadrature}
    eveness = (0,1,-1)
    n_arr = np.arange(0,n)
    a_n=np.zeros(n)
    b_n=np.zeros(n)
    if method not in methods.keys():
        raise ValueError(f"Passed method not found, select method from one of {methods} ")
    if even_flag not in eveness:
        raise ValueError(f"Passed even_flag not found, select flag value from {eveness} ")

    inte = methods[method]
    print(inte)

    calc_a_n = np.vectorize(lambda k: (1/L)*inte(lambda x : np.cos(k*np.pi*x/L)*func(x),a=0,b=2*L))
    calc_b_n = np.vectorize(lambda k: (1/L)*inte(lambda x : np.sin(k*np.pi*x/L)*func(x),a=0,b=2*L))
    
    if even_flag == 0:
        a_n = calc_a_n(n_arr)
        a_n[0] /= 2
    elif even_flag == 1:
        b_n = calc_b_n(n_arr)
    else:
        a_n = calc_a_n(n_arr[:n//2])
        a_n[0] /= 2
        b_n = calc_b_n(n_arr[:n//2+1])

    return(a_n,b_n)

def Partials(func,n,L,d,method="quad",even_flag=-1):
    cosnx = lambda x,i : np.cos(i*np.pi*x/L)
    sinnx = lambda x,i : np.sin(i*np.pi*x/L)

    a_i,b_i = FourierCoeff(func,n,L,d,method,even_flag)
    print(a_i,b_i)
    na = np.arange(0,len(a_i))
    nb = np.arange(1,len(b_i))
    return (np.vectorize(lambda x :np.sum(cosnx(x,na)*(a_i))+np.sum(sinnx(x,nb)*(b_i[1:]))))

if __name__=="__main__":
    from scipy.fft import fft
    g = lambda x:x**2
    f = Partials(lambda x:x**2,10,5,10,even_flag=1)
    
    #print(fft(g(np.linspace(0,2))))
    print(f(0.2),g(0.2))


