
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def Convergence_history(filename=None, save=False):
    
    iter = []
    total_thrust_lst = []
    df = pd.read_csv(filename) if filename else None
    df = df.ffill(axis=1)
    rho = 1.007
    Uinf = 60
    B = 6 # Number of blades
    for i in range (1,len(df.columns)):
        def integrand(x,fx):
            return fx*x
        Ct = df[df.columns[i]]

        fx = Ct*1/2*rho*Uinf**2*np.pi*0.7**2*df[df.columns[0]]
        total_thrust = np.trapezoid(fx, df[df.columns[0]])
        iter.append(i)
        total_thrust_lst.append(total_thrust)

    print(total_thrust_lst[-1])
    plt.plot(iter, total_thrust_lst, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Total Thrust (N)')
    plt.title('Total Thrust vs Iteration')
    plt.grid()
    #Mark the final point
    plt.plot(iter[-1], total_thrust_lst[-1], marker='o', markersize=10, color='red', label={f'Final thrust: {total_thrust_lst[-1]:.2f} N'})
    plt.legend()
    if save:
        plt.savefig('convergence_history.png', dpi=300,)
    plt.show()

    