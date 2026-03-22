import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def alpha_span_dist(save=False, exp_df=None, val_df=None):
    plt.plot(exp_df['r_R'], exp_df['alpha'], '-', label=r'$\alpha$', color='green', linewidth=2)
    plt.plot(exp_df['r_R'], np.rad2deg(exp_df['inflow']), '--', label=r'$\phi$', color='blue', linewidth=2)
    
    if val_df is not None:
        plt.plot(val_df['r_R'], val_df['alpha'], ':', label='Angle of attack (JavaProp)', color='green', marker='o', markersize=4)
        plt.plot(val_df['r_R'], val_df['inflow'], ':', label='Inflow angle (JavaProp)', color='blue', marker='s', markersize=4)
    
    plt.ylabel('Angle [deg]')
    plt.xlabel('(r/R)')
    # plt.title('Alpha and Inflow angle vs Spanwise location')
    plt.legend(loc='upper right')
    plt.grid()
    if save:
        plt.savefig('alpha_span_dist.png')
    plt.show()
    
    
    

# if __name__ == "__main__":
#     exp_df = pd.read_csv('Plots/propeller_radial_data.csv')
#     val_df = pd.read_csv('Plots/validation_data.csv')  # if available
#     alpha_span_dist(save=False, exp_df=exp_df, val_df=val_df)