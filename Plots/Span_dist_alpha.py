import pandas as pd
import matplotlib.pyplot as plt
def alpha_span_dist(save=False):
    df = pd.read_csv('Plots/BEM_randomdata.csv')
    plt.plot( df['r_R'],df['alpha'], label='Angle of attack',color = 'pink')
    plt.plot( df['r_R'],df['inflow'], '--', label='Inflow angle',color = 'blue')
    plt.ylabel('Alpha and inflow angle')
    plt.xlabel('Spanwise location (r/R)')
    plt.title(' Alpha and Inflow angle vs Spanwise location')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    if save:
        plt.savefig('alpha_span_dist.png')

if __name__ == "__main__":
    alpha_span_dist(save=False)

