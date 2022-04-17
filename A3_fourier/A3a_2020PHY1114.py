from Myintegration import *

def FourierCoeff(func,n,L,d=None,method="quad",even_flag=-1,hr_flag = 0):
    methods = {"trap":MyTrap,"simp":MySimp,"quad":MyLegQuadrature}
    eveness = (0,1,-1)
    if method not in methods.keys():
        raise ValueError(f"Passed method not found, select method from one of {methods} ")
    if even_flag not in eveness:
        raise ValueError(f"Passed even_flag not found, select flag value from {eveness} ")
    
    n_arr = np.arange(0,n)
    a_n=np.zeros(n)
    b_n=np.zeros(n)
    inte = methods[method]

    uppr_limit = 2*L
    coef_div = L
    if hr_flag == 1:
        uppr_limit/=2
        coef_div /= 2

    calc_a_n = np.vectorize(lambda k: (1/coef_div)*inte(lambda x : np.cos(k*np.pi*x/L)*func(x),
    a=0,b=uppr_limit,d=d)[0])
    calc_b_n = np.vectorize(lambda k: (1/coef_div)*inte(lambda x : np.sin(k*np.pi*x/L)*func(x),
    a=0,b=uppr_limit,d=d)[0])
    
    if even_flag == 0 or even_flag ==-1:
        a_n = calc_a_n(n_arr)
    if even_flag == 1 or even_flag ==-1:
        b_n = calc_b_n(n_arr)

    a_n[0] /= 2
    return(a_n,b_n)

def Partials(func,n,L,d,method="quad",even_flag=-1,hr_flag=0):
    cosnx = lambda x,i : np.cos(i*np.pi*x/L)
    sinnx = lambda x,i : np.sin(i*np.pi*x/L)

    a_i,b_i = FourierCoeff(func,n,L,d,method,even_flag,hr_flag)
    print(a_i,b_i)
    na = np.arange(0,len(a_i))
    nb = np.arange(1,len(b_i))
    return (np.vectorize(lambda x :np.sum(cosnx(x,na)*(a_i))+np.sum(sinnx(x,nb)*(b_i[1:]))))

def main():
    terms_arr = [1,2,5,10,20]
    def p1(x):
        if -1<=x and x<=0:
            return(0)
        elif 0<=x and x<=1:
            return(1)
        elif x>1 :
            return(p1(x-2))
        elif x<-1 :
            return(p1(x+2))
    def p2(x):
        if -1<=x and x<=-0.5:
            return(0)
        elif -0.5<=x and x<=0.5:
            return(1)
        elif 0.5<=x and x<=1:
            return(0)
        elif x>1 :
            return(p2(x-2))
        elif x<-1 :
            return(p2(x+2))
    def p3(x):
        if -1<=x and x<=0:
            return(-0.5)
        elif 0<=x and x<=1:
            return(0.5)
        elif x>1 :
            return(p3(x-2))
        elif x<-1 :
            return(p3(x+2))
    p1 = np.vectorize(p1)
    p2 = np.vectorize(p2)
    p3 = np.vectorize(p3)
    #p1 = np.vectorize(lambda x: np.piecewise(x,[-1<=x and x<0,0<=x and x<=1],[lambda x: 0,lambda x: 1 ]))
    #p2 = np.vectorize(lambda x: np.piecewise(x,[-1<=x and x<-0.5,-0.5<=x and x<0.5,0.5<=x and x<=1],[lambda x: 0,lambda x: 1,lambda x: 0 ]))
    #p3 = np.vectorize(lambda x: np.piecewise(x,[-1<=x and x<0,0<=x and x<=1],[lambda x: -0.5,lambda x: 0.5 ]))

    x_space = np.linspace(-2.5,2.5,300)
    cols = ["x","f(x)"]
    table1 = np.array([-0.5,0,0.5])
    table2 = table1.copy()
    table3 = table1.copy()
    
    fig1,ax1 = plt.subplots(1,1)
    plt.plot(x_space,p1(x_space),label= "$f(x)$")
    table1 = np.column_stack((table1,p1(table1))) 
    for terms in terms_arr:
        cols.extend(["$S_{"+f"{terms}"+"}$","$RE_{"+f"{terms}"+"}$"])
        f1 = Partials(p1,n=terms,L=1,d=7,method="quad",even_flag=-1)
        table1 = ad2table(table1,f1)
        plt.plot(x_space,f1(x_space),label= "$S_{"+f"{terms}"+"}$")
    ax1.legend()

    
    fig2,ax2 = plt.subplots(1,1)
    plt.plot(x_space,p2(x_space),label= "$f(x)$")
    table2 = np.column_stack((table2,p2(table2))) 

    for terms in terms_arr:
        f2 = Partials(p2,n=terms,L=1,d=7,method="quad",even_flag=0)
        table2 = ad2table(table2,f2)
        plt.plot(x_space,f2(x_space),label= "$S_{"+f"{terms}"+"}$")
    ax2.legend()
    
    fig3,ax3 = plt.subplots(1,1)
    plt.plot(x_space,p3(x_space),label= "$f(x)$")    
    table3 = np.column_stack((table3,p3(table3))) 

    for terms in terms_arr:
        f3 = Partials(p3,n=terms,L=1,d=7,method="quad",even_flag=-1)
        table3 = ad2table(table3,f3)
        plt.plot(x_space,f3(x_space),label= "$S_{"+f"{terms}"+"}$")
    ax3.legend()

    dfs = []
    for i in range(1,4) :
        dfs.append(pd.DataFrame(eval(f"table{i}"),columns=cols))
        dfs[i-1].to_csv(f"p{i}.csv")
        print(dfs[i-1])
        
    plt.show()

def ad2table(t,f):
    t = np.column_stack((t,f(t[:,0])))
    t = np.column_stack((t,(f(t[:,0])-t[:,1])/t[:,1]))
    return t

if __name__=="__main__":
    from scipy.fft import fft
    import matplotlib.pyplot as plt
    from matplotlib import use
    import pandas as pd
    use("WebAgg")
    plt.style.use("bmh")
    main()
