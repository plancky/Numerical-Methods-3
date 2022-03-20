import numpy as np
from Myintegration import *
import matplotlib.pyplot as plt

plt.style.use("bmh")

delta_lorr = np.vectorize(lambda x,e,a=0: e/(((x-a)**2 + e**2)*np.pi))
delta_sin = np.vectorize(lambda x,e,a=0: np.sin((x-a)/e)/((x-a)*np.pi))
delta_decay1 = np.vectorize(lambda x,e,a=0: np.exp(-np.abs((x-a))/e)/(2*e))
delta_decay2 = np.vectorize(lambda x,e,a=0: np.exp(-(x-a)**2/(4*e))/(2*np.sqrt(e*np.pi)))
delta_sech = np.vectorize(lambda x,e,a=0: 1/(np.cosh((x-a)/e)*2*e) )

def f_e(f,e):
    return(lambda x,a=0: f(x,e,a))
 
x_space = np.linspace(-1,1,200)

def makePlots(f,x_space,f_title = "" ):
    dat= np.zeros((3,2)) 
    herm_vardat = np.zeros((3,10))
    herm_nvar = 2**np.arange(1,11) 
    fig,ax = plt.subplots(1,1)
    e_arr = 0.4*2**(-1*np.arange(1,6,1,dtype=float))
    for e in e_arr:
        y = f_e(f,e)(x_space)
        plt.plot(x_space,y,label="$\epsilon$="+str(e))
    print(f_title)

    f_lst1 = [f_e(f,e) ,lambda x: f_e(f,e)(x)*(x+1)**2,lambda x: f_e(f,e)(3*x+1)*9*x**2]
    dat[:,0] = MyLegQuadrature(f_lst1,-1e5,1e5,n=80,m=int(2e3))
    dat[:,1] = MyHermiteQuad(f_lst1,n=80)
    ##
    np.savetxt(f"1114_{f_title}.dat",dat)
    plt.legend()
'''
def get_hermdat(f,plot = False):
    herm_vardat = np.zeros((3,10))
    herm_nvar = 2**np.arange(1,11)
    fig,ax = plt.subplots(1,1)

    f_lst1 = [f_e(f,e) ,lambda x: f_e(f,e)(x)*(x+1)**2,lambda x: f_e(f,e)(3*x+1)*9*x**2]
    for i,igrand in enumerate(f_lst1):
        herm_vardat[i,:] = MyHermiteQuad(igrand,herm_nvar)
        if plot:
            plt.plot()
'''

def get_fi(f,rtol=0.5*10**(-5)):
    e_arr = 0.4*2**(-1*np.arange(2,5,1,dtype=float))
    herm_nvar = 2**np.arange(1,9)
    leg_nvar = 2**np.arange(5,10)

    #dat = np.zeros((len(e_arr),len(herm_nvar)))
    dat = np.zeros((len(e_arr),len(leg_nvar)))
    for i,e in enumerate(e_arr):
        #dat[i] = MyHermiteQuad(lambda x: f(x,e),herm_nvar)
        dat[i] = MyLegQuadrature(lambda x: f(x,e),-1e5,1e5,leg_nvar,m=int(2e3))
        if i == 0 :
            continue
        if dat[i-1][1]-dat[i][1] <= rtol*dat[i][1]:
            np.savetxt(f"herm_vare_",dat)
            print("Tolerance reached ----------------------------------------")
    print(dat)
    plt.plot(e_arr,dat[:,-1],label= f"err_")
    #plt.plot(herm_nvar,dat[-1],label= f"err_",marker ="1")
    return(dat)

#get_fi= np.vectorize(get_fi,excluded=[1,])

def get_edat(f,rtol=0.5*10**(-5)):
    fig,ax = plt.subplots(1,1)
    f_lst1 = [lambda x,e:f_e(f,e)(x) ,lambda x,e: f_e(f,e)(x)*(x+1)**2,lambda x,e: f_e(f,e)(3*x+1)*9*x**2]

    for g in f_lst1[0:1]:
        get_fi(g)

    return(0)




#makePlots(delta_lorr,x_space,f_title = "delta_lorr")
#makePlots(delta_sin,x_space,f_title= "delta_sin")
#makePlots(delta_sech,x_space,f_title= "delta_sech")
#makePlots(delta_decay1,x_space,"exp_decay")
#makePlots(delta_decay2,x_space,"gaussian_distri")
get_edat(delta_lorr)
#get_edat(delta_decay2)

plt.show()
