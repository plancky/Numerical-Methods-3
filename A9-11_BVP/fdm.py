from A11 import *
import matplotlib.pyplot as plt
from matplotlib import use
use("webAgg")
plt.style.use("bmh")

def regress(x,y):
    return np.linalg.lstsq(np.vstack([x,np.ones(x.shape)]).T,y,rcond=None)[0]

bvp1 = ordinary_bvp(lambda x: np.pi**2,lambda x:0,lambda x:-2*np.pi**2*np.sin(np.pi*x),(0,1),lambda x : np.sin(np.pi*x))
#xi , ynumi , yexacti , Ei = |yexacti − ynumi |
bvp1.discretize(1)
bvp1.set_robin((1,0,0),(1,0,0))
fig,ax = plt.subplots(1,1)
N = 2**np.arange(1,8)
for i,ni in enumerate(N):
    bvp1.discretize(ni)
    bvp1.solve()
    bvp1.plot_num(ax)
bvp1.plot_exact(ax)
ax.legend()
plt.show()
'''fig,ax = plt.subplots(1,1)
print(bvp1.lnE(ax))
ax.legend()
fig.savefig("lnE.png")
plt.show()'''
'''N = 2**np.arange(1,7,dtype=int)
rmse = np.zeros(N.shape)
mabse = rmse.copy() 
print(N)
for i,ni in enumerate(N):
    bvp1.discretize(ni)
    bvp1.set_robin((1,0,0),(1,0,0))
    soln,rmse[i],abserr = bvp1.solve()
    mabse[i]= np.max(abserr)
datE = np.array([N,rmse,mabse]).T
datE[:,1:] = np.log(datE[:,1:])
np.savetxt("lnE_bvp1.csv",datE,delimiter=",",header="# N, ln(rmse),ln(abserr)")
m1,c1 = regress(N,datE[:,1])
m2,c2 = regress(N,datE[:,2])
print(m1,m2)'''
    
"""N = 3
dat = np.zeros((N+1,5))
bvp1.discretize(N)
dat[:,0] = np.arange(N+1)
dat[:,1],dat[:,3] = bvp1.ddom,bvp1.exact(bvp1.ddom)
A,b = bvp1.get_A_b()
dat[:,2],rms,dat[:,4] = bvp1.solve()
np.savetxt(f"datE_fdm_N{N}.csv",dat,delimiter=",",fmt="%.6g",header=r"$i$, $x_i$ , $y^{num}_i$ ,$ y^{exact}_i$ ,$ E_i = |y^{exact}_i - y^{num}_i|$")
#fig1,ax1 = plt.subplots(1,1)
#bvp1.plot_exact(ax1)
#bvp1.plot_num(ax1)
#plt.legend()
#plt.show()"""

y2_exact = lambda x : 3/8*np.sin(x) - np.cos(x) - 1/8*np.sin(3*x) 
bvp2 = ordinary_bvp(lambda x: -1,lambda x:0,lambda x:np.sin(3*x),(0,np.pi/2),y2_exact)
'''bvp2.discretize(1)
bvp2.set_robin((1,1,-1),(0,1,1))
fig,ax = plt.subplots(1,1)
N = 2**np.arange(1,8)
for i,ni in enumerate(N):
    bvp2.discretize(ni)
    bvp2.set_robin((1,1,-1),(0,1,1))
    bvp2.solve()
    bvp2.plot_num(ax)
bvp2.plot_exact(ax)
ax.legend()
plt.show()'''

'''
N = 8
dat = np.zeros((N+1,5))
bvp2.discretize(N)
dat[:,0] = np.arange(N+1)
dat[:,1],dat[:,3] = bvp2.ddom,bvp2.exact(bvp2.ddom)
bvp2.set_robin((1,1,-1),(0,1,1))
A,b = bvp2.get_A_b()
dat[:,2],rms,dat[:,4] = bvp2.solve()
np.savetxt(f"datE_fdm_bvp2_N{N}.csv",dat,delimiter=",",fmt="%.6g",header=r"$i$, $x_i$ , $y^{num}_i$ ,$ y^{exact}_i$ ,$ E_i = |y^{exact}_i - y^{num}_i|$")
A,b = bvp2.get_A_b()'''
