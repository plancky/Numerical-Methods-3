from fourier_legendre import *
import numpy as np
from scipy.special import eval_legendre
import matplotlib.pyplot as plt

def main():
    part1 = lambda x: 2*x**4 + 3*x + 2
    part1_coefs = np.round_(legcoeff(part1,5),14)
    print(part1_coefs[part1_coefs!=0],np.where(part1_coefs!=0)[0]) 

    part2 = lambda x : np.cos(x)*np.sin(x)
    x_space = np.linspace(-np.pi,np.pi)
    
    leg_apr_f = fourier_leg(part2,1)
    y_apr_0 = leg_apr_f(x_space)
    for n_i in np.arange(2,10000):
        leg_apr_f = fourier_leg(part2,n_i)
        y_apr_1 = leg_apr_f(x_space)
        err = max(abs(y_apr_1-y_apr_0))
        rms = np.sqrt(np.sum((y_apr_1-y_apr_0)**2))/y_apr_0.shape[0]
        print(f"Max difference between values of f_{n_i}-f_{n_i-1}:",err)
        print(f"RMS ERROR between values of f_{n_i}-f_{n_i-1}:",rms)
        if err <= 0.5/10**6 * np.max(np.abs(y_apr_1)):
            print("Accuracy of 6 significant digits reached with number of terms equal to:",n_i-1)
            break
        y_apr_0 = y_apr_1.copy()

    
    x_space = np.linspace(-2,2)
    fig,(ax1,ax2) = plt.subplots(1,2)
    nplt1 = np.arange(1,6)
    ax1.plot(x_space,part1(x_space),marker='1',label="$f(x)$")
    for n_i in nplt1:
        leg_apr = fourier_leg(part1,n_i)
        ax1.plot(x_space,leg_apr(x_space),label=f"$f_{n_i}$",marker="1")
    ax1.set_title("Function Series plots for $f(x) =2x^4 + 3x + 2 $ ")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$f(x)$")
    nplt2 = np.arange(2,12,2)
    ax2.plot(x_space,part2(x_space),marker='1',label="$f(x)$")
    for n_i in nplt2:
        leg_apr = fourier_leg(part2,n_i)
        ax2.plot(x_space,leg_apr(x_space),label="$f_{"+f"{n_i}"+"}$",marker="1")
    ax2.set_title("Function Series plots for $f(x) =\sin(x)\cos(x)$ ")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$f(x)$")
    ax1.legend()
    ax2.legend()
    plt.show()
    

if __name__ =="__main__":
    plt.style.use("bmh")
    #from matplotlib import use
    #use("WebAgg")
    main()