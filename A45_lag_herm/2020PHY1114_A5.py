from Myintegration import *

#
# Validation
#
p1 = lambda x : (x**4 + x**2) * np.exp(-x**2) 
p2 = lambda x : (x**3 + x) * np.exp(-x**2)
p3 = lambda x : (x**8 + x**2) * np.exp(-x**2) 

print("Numerical : ",MyHermiteQuad(p1,n=2)," Analytical : 1.34329342164673517043712359...")
print("Numerical : ",MyHermiteQuad(p2,n=2)," Analytical : ",np.pi)

######
I1 = lambda x : np.exp(-x**2)/(1+x**2)
I2 = lambda x : 1/(1+x**2)

print("Numerical : ",MyHermiteQuad(I1,n=50)," Analytical : 1.34329342164673517043712359...")
print("Numerical : ",MyHermiteQuad(I2,n=50)," Analytical : ",np.pi)

dat = np.zeros((7,3))
dat[:,0] = 2**np.arange(1,8)
n_arr = dat[:,0]
dat[:,1] = MyHermiteQuad(I1,n_arr)
dat[:,2] = MyHermiteQuad(I2,n_arr)

def simpsonHermite(f,max_b= int(1e6),rtol = 0.5*10**(-4)):
    b = 10**np.arange(1,np.floor(np.log10(max_b)))
    I = np.zeros(b.shape)
    for i,bi in enumerate(b):
        I[i] = MySimp(f,-bi,bi,m = int(1e7),d =5)[0]
        if i == 0:
            continue
        if np.abs(I[i]-I[i-1])<=rtol*I[i]:
            return(I[i],bi)
    return(I[-1],b[-1])

print(simpsonHermite(I1))
print(simpsonHermite(I2))
np.savetxt("quad-herm-1114.out",dat,fmt="%.7g",delimiter=",",header="n,$I_1$,$I_2$")
