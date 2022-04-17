from Myintegration import *

#
# Validation
#

p1 = lambda x : (x**4 + x**2) * np.exp(-x) 
p2 = lambda x : (x**3 + x) * np.exp(-x)
p3 = lambda x : (x**8 + x**2) * np.exp(-x) 

print("Numerical : ",MyLeguQuad(p1,n=2)," Analytical : ",26)
print(MyLeguQuad(p2,n=2))

######
I1 = lambda x : np.exp(-x)/(1+x**2)
I2 = lambda x : 1/(1+x**2)

print(MyLeguQuad(I1,n=50))
print(MyLeguQuad(I2,n=55))

dat = np.zeros((7,3))
dat[:,0] = 2**np.arange(1,8)
n_arr = dat[:,0]
dat[:,1] = MyLeguQuad(I1,n_arr)
dat[:,2] = MyLeguQuad(I2,n_arr)

def simpsonLagu(f,max_b= int(1e8),rtol = 0.5*10**(-4)):
    b = 10**np.arange(1,np.floor(np.log10(max_b)))
    I = np.zeros(b.shape)
    for i,bi in enumerate(b):
        I[i] = MySimp(f,0,bi,m = int(1e7),d =5)[0]
        if i == 0:
            continue
        if np.abs(I[i]-I[i-1])<=rtol*I[i]:
            return(I[i],bi)
    return(I[-1],b[-1])

print(simpsonLagu(I1))
print(simpsonLagu(I2))
np.savetxt("quad-lag-1114.out",dat,fmt="%.7g",delimiter=",",header="n,$I_1$,$I_2$")
