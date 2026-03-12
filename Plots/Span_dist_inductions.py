import pandas as pd
import matplotlib.pyplot as plt
def induction_span_dist(save=False):
    df = pd.read_csv('Plots/BEM_randomdata.csv')
    plt.plot( df['r_R'],df['a'], label='axial induction factor',color = 'green')
    plt.plot( df['r_R'],df['a_prime'], '--', label='tangential induction factor',color = 'orange')
    plt.ylabel('Induction factors')
    plt.xlabel('Spanwise location (r/R)')
    plt.title(' Induction factors vs Spanwise location')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    if save:
        plt.savefig('induction_factors_span_dist.png')

if __name__ == "__main__":
    induction_span_dist(save=False)