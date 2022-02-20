from fourier_legendre import *
import numpy as np
from scipy.special import eval_legendre
import matplotlib.pyplot as plt

def main():
    part1 = lambda x: 2*x**4 + 3*x + 2
    #print([f"{d:.6g}" for d in legcoeff(part1,5,8)])

    part2 = lambda x : np.cos(x)*np.sin(x)
    #print([float(f"{d:.6g}") for d in legcoeff(part2,10,8)])
    f_approx = np.vectorize(partialseries)
    '''
    x_space = np.linspace(-1,1)
    y_apr_0 = f_approx(part2,x_space,1)
    for n_i in np.arange(2,10000):
        y_apr_1 = f_approx(part2,x_space,n_i)
        err = max(abs(y_apr_1-part2(x_space)))
        print(err)
        if err <= 0.5/10**6 * np.max(np.abs(y_apr_1)):
            print(n_i)
            break'''
    x_space = np.linspace(-2,2)
    fig,(ax1,ax2) = plt.subplots(1,2)
    nplt1 = np.arange(1,6)
    ax1.plot(x_space,part1(x_space),marker='1',label="$f(x)$")
    for n_i in nplt1:
        ax1.plot(x_space,f_approx(part1,x_space,n_i),label=f"$f_{n_i}$",marker="1")
    ax1.set_title("Function Series plots for $f(x) =2x^4 + 3x + 2 $ ")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$f(x)$")
    nplt2 = np.arange(2,12,2)
    ax2.plot(x_space,part2(x_space),marker='1',label="$f(x)$")
    for n_i in nplt2:
        ax2.plot(x_space,f_approx(part2,x_space,n_i),label="$f_{"+f"{n_i}"+"}$",marker="1")
    ax2.set_title("Function Series plots for $f(x) =\sin(x)\cos(x)$ ")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$f(x)$")
    
    
    ax1.legend()
    ax2.legend()
    plt.show()
    #for n_i in np.arange()
    #print(f_approx(part2,x_space,n)- part2(x_space))


if __name__ =="__main__":
    plt.style.use("bmh")
    main()