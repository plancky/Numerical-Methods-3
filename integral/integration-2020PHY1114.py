from Myintegration import *
import matplotlib.pyplot as plt
import pandas as pd

def main():
    n = 2*np.arange(1,17)
    h = 1/n
    #f_str = input("Enter the function : ")
    #f = lambda x :eval(f_str)
    f = lambda x : 1/(1+x**2)
    my_pi_simp = 4*MySimp(f,0,1,n)
    my_pi_trap = 4*MyTrap(f,0,1,n)

    err_simp= np.abs(my_pi_simp-np.pi) 
    err_trap= np.abs(my_pi_trap-np.pi)
     
    #fig,(ax1,ax2) = plt.subplots(1,2,1)
    signi_digits = 5
    tab_dat = np.array([MyTrap(f,0,1,10000,signi_digits),MySimp(f,0,1,10000,signi_digits)])
    
    pi_arr =np.pi*np.ones((2,))
    #my_pi_simp_6,n_simp_6 = 4*MySimp(f,0,1,10000,signi_digits)
    #my_pi_trap_6,n_trap_6 = 4*MySimp(f,0,1,10000,signi_digits)
    df = pd.DataFrame({"Method":["Trapezoidal","Simpson1/3"],"Pi_calc":4*tab_dat[:,0], "n":tab_dat[:,1],"E" : np.abs(4*tab_dat[:,0] - pi_arr)/pi_arr})
    print(df)
    
    '''
    plt.plot(n,np.arccos(-1)*np.ones(n.shape),label = "")
    plt.plot(n,my_pi_simp,label= "my_pi_simp(n)")
    plt.plot(n,my_pi_trap,label= "my_pi_trap(n)")
    plt.plot()
    plt.show()
    plt.plot(n,err_simp,label= "e_simp(n)")
    plt.plot(n,err_trap,label= "e_trap(n)")
    plt.plot()
    plt.show()
    plt.plot(np.log(h),np.log(err_trap),label= "e_trap(n)")
    plt.plot(np.log(h),np.log(err_simp),label= "e_simp(n)")
    '''


if __name__ =="__main__":
    plt.style.use("ggplot")
    main()
    plt.plot()
    plt.legend()
    plt.show()