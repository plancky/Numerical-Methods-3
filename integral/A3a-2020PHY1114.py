from sympy import integrate
from Myintegration import *


def FourierCoeff(func,n,L,d,method="quad",even_flag=-1):
    methods = {"trap":MyTrap,"simp":MySimp,"quad":MyLegQuadrature}
    eveness = (0,1,-1)
    n_arr = np.arange(0,n)
    a_n=np.zeros(n)
    b_n=np.zeros(n)
    if method not in methods.keys():
        raise ValueError(f"Passed method not found, select method from one of {methods} ")
    if even_flag not in eveness:
        raise ValueError(f"Passed even_flag not found, select flag value from {eveness} ")

    integrate = methods[method]

    calc_a_n = np.vectorize(lambda n: integrate(lambda x : np.cos(n*np.pi*x/L)*func(x),-L,L,d=int(d))[0])
    calc_b_n = np.vectorize(lambda n: integrate(lambda x : np.sin(n*np.pi*x/L)*func(x),-L,L,d=int(d))[0])
    
    if even_flag == 0:
        a_n = calc_a_n(n_arr)
        a_n[0] /= 2
    elif even_flag == 1:
        b_n = calc_b_n(n_arr)
    else:
        a_n = calc_a_n(n_arr[:n//2])
        b_n = calc_b_n(n_arr[:n//2+1])

    return(a_n,b_n)

def Partials(func,n,L,d,method="quad",even_flag=-1):
    cosnx = lambda x,n : np.cos(n*np.pi*x/L)
    sinnx = lambda x,n : np.sin(n*np.pi*x/L)

    a_i,b_i = FourierCoeff(func,n,L,d,method,even_flag)
    print(a_i)
    na = np.arange(0,len(a_i))
    nb = np.arange(0,len(b_i))
    return (np.vectorize(lambda x :cosnx(x,na).dot(a_i)+sinnx(x,nb).dot(b_i)))

if __name__=="__main__":
    g = lambda x:x**2
    f = Partials(lambda x:x**2,5,1,2,even_flag=0)
    print(f(0.5),g(0.5))

