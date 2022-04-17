from A3a_2020PHY1114 import *


def main():
    def p4(x):
        if 0<=x and x<=np.pi:
            return(x) 
        
    p4 = np.vectorize(p4)
    terms_arr = [1,2,5,10,20]

    x_space = np.linspace(-3*np.pi,3*np.pi,300)
    
    cols = ["x","f(x)"]
    table1 = np.array([0,np.pi/2,np.pi])
    table2 = table1.copy()

    fig1,ax1 = plt.subplots(1,1)
    ax1.plot(x_space,p4(x_space),label= "$f(x)$")
    table1 = np.column_stack((table1,p4(table1))) 
    for terms in terms_arr:
        cols.extend(["$S_{"+f"{terms}"+"}$","$RE_{"+f"{terms}"+"}$"])
        f4 = Partials(p4,n=terms,L=np.pi,d=7,method="quad",even_flag=0,hr_flag=1)
        table1 = ad2table(table1,f4)
        ax1.plot(x_space,f4(x_space),label= "$S_{"+f"{terms}"+"}$",marker="1")
    ax1.legend()

    fig2,ax2 = plt.subplots(1,1)
    ax2.plot(x_space,p4(x_space),label= "$f(x)$")
    table2 = np.column_stack((table2,p4(table2))) 
    for terms in terms_arr:
        f4 = Partials(p4,n=terms,L=np.pi,d=7,method="quad",even_flag=1,hr_flag=1)
        table2 = ad2table(table2,f4)
        ax2.plot(x_space,f4(x_space),label= "$S_{"+f"{terms}"+"}$",marker="1")
    ax2.legend()
    
    dfs = []
    for i in range(1,3) :
        dfs.append(pd.DataFrame(eval(f"table{i}"),columns=cols))
        dfs[i-1].to_csv(f"h{i}.csv")
        print(dfs[i-1])
    
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import use
    import pandas as pd
    use("WebAgg")
    plt.style.use("bmh")
    main()





