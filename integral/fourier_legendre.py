from Myintegration import *
from scipy.special import legendre
from scipy.special import eval_legendre

def inp(f1,f2,a,b,n):
    return(MyLegQuadrature(lambda x: f1(x)*f2(x),a,b,n))

def legcoeff(func,n):
    n_arr = np.arange(0,n)
    coeffs = np.vectorize(lambda n: (2*n+1)/2 * inp(legendre(n),func,-1,1,6))
    return(coeffs(n_arr))

def fourier_leg(f,n):
    coeff = legcoeff(f,n)
    n_arr = np.arange(0,len(coeff))
    return(np.vectorize(lambda x: eval_legendre(n_arr,x).dot(coeff)))

if __name__ == "__main__":
    print([f"{d:.8g}" for d in legcoeff(lambda x: 2*x**4 + 3*x + 2,5,8)])