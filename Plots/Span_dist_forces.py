
import pandas as pd
import matplotlib.pyplot as plt
def forces_span_dist(save=False):
    df = pd.read_csv('Plots/BEM_randomdata.csv')
    plt.plot( df['r_R'],df['Cx'], label='axial force/thrust', color='yellow')
    plt.plot( df['r_R'],df['Cy'], '--', label='tangential force/azimutahl loading', color='red')
    plt.ylabel('Force and torque')
    plt.xlabel('Spanwise location (r/R)')
    plt.title(' axial and aziumuthal loading vs Spanwise location')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    if save:
        plt.savefig('forces_span_dist.png')

if __name__ == "__main__":
    forces_span_dist(save=False)