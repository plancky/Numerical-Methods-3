import numpy as np
from numpy import vectorize as vec
def get_tridiag(l,d,u):
    N = len(d)
    if not (N-1 == len(l) and N-1 ==len(u)):
        raise ValueError(f"length of l and d must be {N-1}.")
    return np.diag(l,-1) + np.diag(d,0) + np.diag(u,1)

#y′′ (x) + p(x)y ′ (x) + q(x)y(x) + r(x) = 0
class ordinary_bvp:
    def __init__(self,p,q,r,dom=(0,1)) -> None:
        self.p = p 
        self.q = q 
        self.r = r
        self.dom = dom
        pass

    def discretize(self,N:int):
        self.N = N 
        self.ddom = np.linspace(self.dom[0],self.dom[1],N+1)
        self.w = np.zeros(self.ddom.shape) 
        #ddom,w = [0,1,2....N-1,N]
        self.h = abs(np.diff(self.ddom)[0])/N
        h,x = self.h,self.ddom
        self.d = vec(lambda i : 2 + h**2*self.p(x[i]))
        self.u = vec(lambda i : -1 + h/2*self.q(x[i]))
        self.l = vec(lambda i : -1 - h/2*self.q(x[i]))
        self.b = vec(lambda i : -h**2*self.r(x[i]))  
        return self.ddom

    def set_a(self,c,btype = "nn"):
        h,x = self.h,self.ddom
        b_,d,l,u,N = self.b,self.d,self.l,self.u,self.N
        if btype == "diri":
            self.a11,self.a12 = 1,0
            self.b1 = c
        elif btype == "nn":
            self.a11,self.a12 = d(0),-2
            self.b1 = b_(0)+2*h*l(0)*c
        elif btype == "rob":
            (c1,c2,c3) = c
            self.a11,self.a12 = d(0) + 2*h*l(0)*c1/c2,-2
            self.b1= b_(0)+2*h*l(0)*c3/c2
        else:
            raise ValueError("btype should be 'diri','nn' or 'rob'")    
        
    def set_b(self,c,btype = "nn"):
        h,x = self.h,self.ddom
        b_,d,l,u,N = self.b,self.d,self.l,self.u,self.N
        if btype == "diri":
            self.ann,self.an_1n = 1,0
            self.bn = c
        elif btype == "nn":
            self.ann,self.an_1n = d(N),-2
            self.bn = b_(N)-2*h*l(N)*c
        elif btype == "rob":
            (c1,c2,c3) = c
            self.ann,self.an_1n = d(N) - 2*h*u(N)*c1/c2 ,-2
            self.b1= b_(0)+2*h*l(0)*c3/c2
        else:
            raise ValueError("btype should be 'diri','nn' or 'rob'")

    def set_dirichlet(self,a,b):
        self.w[0],self.w[-1] = a,b
        h,x = self.h,self.ddom
        self.a11,self.a12 = 1,0
        self.ann,self.an_1n = 1,0
        self.b1,self.bn = a,b
        return self.w

    def set_neumann(self,a,b):
        self.w[0],self.w[-1] = a,b
        h,x = self.h,self.ddom
        b_,d,l,u,N = self.b,self.d,self.l,self.u,self.N
        self.a11,self.a12 = d(0),-2
        self.ann,self.an_1n = d(N),-2
        self.b1,self.bn = b_(0)+2*h*l(0)*a,b_(N)-2*h*l(N)*b
        return self.w
    
    def set_robin(self,a,b): # a1 y(0) + a2 y'(0) = a3 ; b1 y(N) + b2y'(N) = b3 
        (a1,a2,a3),(b1,b2,b3) = a,b
        h,x = self.h,self.ddom
        b_,d,l,u,N = self.b,self.d,self.l,self.u,self.N
        self.a11,self.a12 = d(0) + 2*h*l(0)*a1/a2,-2
        self.ann,self.an_1n = d(N) - 2*h*u(N)*b1/b2 ,-2
        self.b1,self.bn = b_(0)+2*h*l(0)*a3/a2, b_(N)-2*h*l(N)*b3/b2
        return self.w
        
    def get_A_b(self):
        ii = np.arange(1,self.N)
        l_,d_ = np.zeros(self.N),np.zeros(self.N+1)
        u_,b_ = l_.copy(),d_.copy()
        l_[:-1],l_[-1] = self.l(ii), self.an_1n 
        u_[1:],u_[0] = self.u(ii), self.a12 
        d_[1:-1],d_[0],d_[-1] = self.d(ii),self.a11,self.ann
        b_[1:-1],b_[0],b_[-1] = self.b(ii),self.b1,self.bn 
        A = get_tridiag(l_,d_,u_)
        return A,b_

if __name__ == "__main__":
    bvp1 = ordinary_bvp(lambda x: np.pi**2,lambda x:0,lambda x:-2*np.pi**2*np.sin(np.pi*x),(0,1))
    bvp1.discretize(100)
    bvp1.set_a(0,btype="diri")
    bvp1.set_b(0,btype="diri")
    y1_exact = lambda x : np.sin(np.pi*x) 
    A,b = bvp1.get_A_b()
    print(np.linalg.solve(A,b) - y1_exact(bvp1.ddom))

    bvp2 = ordinary_bvp(lambda x: -1,lambda x:0,lambda x:np.sin(3*x),(0,np.pi/2))
    bvp2.discretize(100)
    bvp2.set_a((1,1,-1),btype="rob")
    bvp2.set_b(1,btype="nn")
    y2_exact = lambda x : 3/8*np.sin(x) - np.cos(x) - 1/8*np.sin(3*x) 
    A,b = bvp2.get_A_b()
    print(np.linalg.solve(A,b) - y2_exact(bvp2.ddom))