from Myintegration import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quadrature
quadrature = np.vectorize(quadrature)

def main():
    #
    # Part(b)
    #    
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
    plt.plot()
    plt.legend()
    plt.show()
    '''
    #
    # Part(d)
    #
    
    '''
    signi_digits = 5
    tab_dat = np.array([MyTrap(f,0,1,int(1e6),signi_digits),MySimp(f,0,1,int(1e6),signi_digits)])    
    pi_arr =np.pi*np.ones((2,))
    df = pd.DataFrame({"Method":["Trapezoidal","Simpson1/3"],"Pi_calc":4*tab_dat[:,0], "n":tab_dat[:,1],"E" : np.abs(4*tab_dat[:,0] - pi_arr)/pi_arr})
    print(df)
    '''
    #
    # Part (e)
    #
    n_points,m_arr = 2**np.arange(1,7),2**np.arange(0,6)
    '''
    nm_mat = np.ones((len(n_points),len(m_arr)))
    for i,n_i in enumerate(n_points):
        nm_mat[i,:] = 4*MyLegQuadrature(f,0,1,n_i,m_arr)
    np.savetxt("pi_quad-1114.dat",nm_mat,delimiter=",",fmt="%.16f")
    '''
    '''
    nm_mat = np.loadtxt("pi_quad-1114.dat",delimiter= ",",dtype=float)
    err_nm_mat = nm_mat - np.arccos(-1)
    print(nm_mat)
    fig,(axs1,axs2) = plt.subplots(1,2)
    axs1.plot(n_points,nm_mat[:,np.where(m_arr==1)[0][0]],label="m=1",marker=".")
    axs1.plot(n_points,nm_mat[:,np.where(m_arr==8)[0][0]],label="m=8",marker=".")
    axs1.plot(n_points,np.arccos(-1)*np.ones(n_points.shape),label = "$\pi = \cos^{-1}(-1)}$")
    axs2.plot(n_points,err_nm_mat[:,np.where(m_arr==1)[0][0]],label="m=1",marker=".")
    axs2.plot(n_points,err_nm_mat[:,np.where(m_arr==8)[0][0]],label="m=8",marker=".")
    axs1.legend();axs2.legend()
    fig1,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(m_arr,nm_mat[np.where(n_points==2)[0][0]],label="n=2",marker="1")
    ax1.plot(m_arr,nm_mat[np.where(n_points==8)[0][0]],label="n=8",marker="1")
    ax1.plot(m_arr,np.arccos(-1)*np.ones(m_arr.shape),label = "$\pi = \cos^{-1}(-1)}$")
    ax2.plot(m_arr,err_nm_mat[np.where(n_points==2)[0][0]],label="n=2",marker="1")
    ax2.plot(m_arr,err_nm_mat[np.where(n_points==8)[0][0]],label="n=8",marker="1")
    ax1.legend();ax2.legend()
    
    #plt.legend()
    plt.show()
    '''
    #
    # Part (f)
    #
    # 
    
    n_points = 2**np.arange(1,6)
    tol_arr = np.arange(2,8)
    #myttype =[('pi',float),('m',float)]
    csv_dat = np.zeros((len(n_points),len(tol_arr)*2))
    #fixed_tol_mat = np.ndarray((len(n_points),len(tol_arr),2),dtype=float) 
    for i,n_i in enumerate(n_points):
        print(n_i)
        tmp = np.column_stack(MyLegQuadrature(f,0,1,n_i,m=10000,d=tol_arr))
        tmp[:,0] *= 4 
        print(tmp) 
        #fixed_tol_mat[i,:]= tmp
        csv_dat[i,:] = tmp.flatten()   
        #print(quadrature(f,0,1,rtol = tol_arr))
    print(csv_dat)
    np.savetxt("f_data.csv",csv_dat,delimiter= ",",fmt="%.18g")
    
if __name__ =="__main__":
    plt.style.use("ggplot")
    #import matplotlib
    #matplotlib.use("WebAgg")
    #%matplotlib
    main()
