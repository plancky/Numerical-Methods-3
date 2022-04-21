from Myintegration import *
from solve_ivp import *
from scipy.optimize import fsolve

func = lambda t,y : np.array([[0,1],[-1,0]]).dot(y)+np.array([0,np.sin(3*t)])

#y1_a = np.linspace(0,1)
y1_b = 1 
y_exact = lambda x:3/8*np.sin(x) - np.cos(x) -1/8*np.sin(3*x)
robin_a = lambda x: -x -1 

def solve_bvp(func,robin,N=100,btype = "nn"):
    def optimize(y0_a):
        p =set_problem(func,[0,np.pi/2],[0,robin(y0_a),y0_a],N)
        o = p.rk4()[-1,:]
        return o[-1]-1
    y0 = fsolve(optimize,0)
    p =set_problem(func,[0,np.pi/2],[0,robin_a(y0),y0],N)
    return(p)

obtained_p = solve_bvp(func,robin_a,N=5)
obtained_p.rk4()
f,ax = plt.subplots(1,1)
obtained_p.jt_plot(ax,1)
x_space = np.linspace(0,np.pi/2)
ax.plot(x_space,y_exact(x_space))
plt.show()