import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def eff_harvesting(save=False):
    U0 = 60
    rho  = 1.007
    df = pd.read_csv('Plots/results.csv')
    
    Pc = df["Total Power"] / (rho * U0**3 * (0.7*2)**2)
    eta_harv = -Pc * 8/np.pi
    # eta_prop = (df["J"] * df["CT"]/df["CP"])

    plt.plot( df['J'],eta_harv, label='harvesting efficiency',color = 'red')
    # plt.plot( df['J'],eta_prop, '--', label='propeller efficiency',color = 'blue')
    plt.ylabel('Efficiency')
    plt.xlabel('Advance ratio J')
    plt.title(' xxx')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    if save:
        plt.savefig('eff_harvesting.png')


if __name__ == "__main__":
    eff_harvesting(save=False)