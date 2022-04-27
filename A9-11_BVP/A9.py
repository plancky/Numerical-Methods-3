from Myintegration import *
from solve_ivp import *
from scipy.optimize import fsolve

func = lambda t,y : np.array([[0,1],[-1,0]]).dot(y)+np.array([0,np.sin(3*t)])

#y1_a = np.linspace(0,1)
y1_b = 1 
y_exact = lambda x:3/8*np.sin(x) - np.cos(x) -1/8*np.sin(3*x)
robin_a = lambda x: -x -1 

def lin_shooting(func,robin,N=100,btype = "nn"):
    def optimize(y0_a):
        p =set_problem(func,[0,np.pi/2],[0,robin(y0_a),y0_a],N)
        o = p.rk4()[-1,:]
        return o[-1]-1
    y0 = fsolve(optimize,0)
    p =set_problem(func,[0,np.pi/2],[0,robin_a(y0),y0],N)
    return(p)

f,ax = plt.subplots(1,1)
x_space = np.linspace(0,np.pi/2)
ax.plot(x_space,y_exact(x_space))
for N in [4,8,16,32,64,128]:
    obtained_p = lin_shooting(func,robin_a,N)
    obtained_p.rk4()
    obtained_p.jt_plot(ax,1)
plt.show()

for N in [4,8]:
    tabl = np.zeros((N+1,4))
    new_p = lin_shooting(func,robin_a,N)
    rk4_soln = new_p.rk4()
    tabl[:,1] = rk4_soln[:,1]
    tabl[:,0] = new_p.dom
    tabl[:,2] = y_exact(tabl[:,0])
    tabl[:,3] = np.abs(tabl[:,1]-tabl[:,2])
    print(tabl)
    np.savetxt(f"dat_E_N{N}.csv",tabl,fmt="%.8E",delimiter=",")
