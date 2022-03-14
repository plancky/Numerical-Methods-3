import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit 
from matplotlib import use
plt.style.use("bmh")
use("WebAgg")
dat= np.loadtxt("data-lsf.csv",delimiter=",")

inptdat = np.zeros((dat.shape[0],3))

inptdat[:,0]= dat[:,0]
inptdat[:,1]= np.mean(dat[:,1:],axis = 1)**2
std = 2*np.sqrt(inptdat[:,1])*np.std(dat[:,1:],axis = 1,ddof=1)/np.sqrt(dat.shape[1]-1)
inptdat[:,2]= 1/(std)**2

'''
for i in range(0,dat.shape[0]): # for each row
    yi = dat[i,1:dat.shape[1]]
    inptdat[i,1]= np.mean(yi)
    inptdat[i,2]= len(yi)/(np.sum((yi-inptdat[i,1])**2)/(len(yi)-1))
'''

def Mywlsf(inpt,weights=False):
    if inpt.shape[1]>3:
        raise ValueError("shape of input matrix must be (N,2) or (N,3) for weighted.")  
    x,y,w = inpt[:,0],inpt[:,1],inpt[:,2]
    if not weights or inpt.shape[1] == 2 :
        w = np.ones(x.shape,dtype=float)
    x_mean,y_mean = np.average(x,0,w),np.average(y,0,w)
    ss_xx = w.dot((x-x_mean)**2) 
    ss_yy = w.dot((y-y_mean)**2)
    ss_xy = w.dot((x-x_mean)*(y-y_mean))
    [ss_x,s_x,s_w] = np.sum([w*x**2,w*x,w],axis=1) 
    delta = (s_w*ss_x - s_x**2)
    a = np.array([ss_xy/ss_xx,y_mean-ss_xy/ss_xx*x_mean])
    resi = y - a[0]*x - a[1]
    r_arr = np.sum([resi,resi**2],axis=1)
    #a = np.array([(s_w*s_xy - s_x*s_y),(ss_x*s_y - s_x*s_xy)])/delta
    if not weights:
        s = np.sqrt(r_arr[1]/(len(x)-2))
        e_a = np.array([s/np.sqrt(ss_xx),s*np.sqrt(1/len(x)+x_mean**2/ss_xx),])
    else :
        e_a = np.sqrt(np.array([s_w,ss_x])/delta)
    corr = [np.sqrt(ss_xy**2/(ss_xx*ss_yy)),(resi**2*w).sum()]
    return(a,e_a,r_arr,corr)

if __name__ == "__main__":
    def get_km(p,e):    
        k= 4*np.pi**2/p[0]
        m = 3*p[1]/p[0] 
        err_k = 4*np.pi**2/p[0]**2 * e[0] 
        err_m = m*np.sqrt((e[0]/p[0])**2 + (e[1]/p[1])**2)
        return(k,m,err_k,err_m)  


    np.savetxt("1114.csv",inptdat)
    se = np.sqrt(1/inptdat[:,2])
    #--------------lsf
    (params2,err2,resi2,corr2) = Mywlsf(inptdat)
    k1,m1,err_k1,err_m1 = get_km(params2,err2)
    print("\nlsf : Parameters => ",params2,"\n Error in params => ",err2,"\n Sum of residuals and sum of square of residuals => ",resi2,"\n corr coef and chi^2 => ",corr2)
    inbuilt = linregress(inptdat[:,0],inptdat[:,1])
    print(inbuilt)
    print("Value of spring constant(k) => ",k1,"+-",err_k1,"\nValue of effective mass of spring(3m) => ",m1,"+-",err_m1)
    #--------------wlsf
    (params,err,resi,corr) = Mywlsf(inptdat,weights=True)
    k,m,err_k,err_m = get_km(params,err)
    print("\nWeighted lsf : Parameters => ",params,"\n Error in params => ",err,"\n Sum of residuals and sum of square of residuals => ",resi,"\n corr coef and chi^2 => ",corr)

    print("Value of spring constant(k) => ",k,"+-",err_k,"\nValue of effective mass of spring(3m) => ",m,"+-",err_m)
    p,pcov = curve_fit(lambda x,a,b:a*x+b,inptdat[:,0],inptdat[:,1],sigma=std ,absolute_sigma=True)
    print("\nInbuilt scipy.optimize.curve_fit => ",p,np.sqrt(np.diag(pcov)))

    #np.savetxt("1114.out",[k,m])
    fig,ax = plt.subplots(1,1)
    ax.errorbar(inptdat[:,0],inptdat[:,1],se,fmt= ".",color='black',
                ecolor='red', elinewidth=3, capsize=0,label = "Data-points, error bars 1STE")
    ax.plot(inptdat[:,0],inptdat[:,0]*params[0]+params[1],c ='cyan',label = "weighted regression line")
    ax.plot(inptdat[:,0],inptdat[:,0]*params2[0]+params2[1],ls = '--',c='green',label = "Ordinary regression line")
    ax.set_xlabel("$M$");ax.set_ylabel("$T^2$");
    ax.legend()
    fig,ax = plt.subplots(1,1)
    ax.scatter(inptdat[:,0],inptdat[:,1],marker= ".",color='black',label = "Data-points")
    ax.plot(inptdat[:,0],inptdat[:,0]*params2[0]+params2[1],c='cyan',label = "Ordinary regression line")
    ax.set_xlabel("$M$");ax.set_ylabel("$T^2$")
    ax.legend()
    plt.legend()
    plt.show()








