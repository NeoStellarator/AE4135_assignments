import pandas as pd
import matplotlib.pyplot as plt

def induction_span_dist(save=False, exp_df=None, val_df=None):
    plt.plot(exp_df['r_R'], exp_df['a'], '-', label='Axial induction (Our model)', color='green', linewidth=2)
    plt.plot(exp_df['r_R'], exp_df['a_prime'], '--', label='Tangential induction (Our model)', color='orange', linewidth=2)
    
    if val_df is not None:
        plt.plot(val_df['r_R'], val_df['a'], ':', label='Axial induction (JavaProp)', color='green', marker='o', markersize=4)
        plt.plot(val_df['r_R'], val_df['a_prime'], ':', label='Tangential induction (JavaProp)', color='orange', marker='s', markersize=4)
    
    plt.ylabel('Induction factors')
    plt.xlabel('Spanwise location (r/R)')
    plt.title('Induction factors vs Spanwise location')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    
    if save:
        plt.savefig('induction_factors_span_dist.png')

# if __name__ == "__main__":
#     exp_df = pd.read_csv('Plots/BEM_randomdata.csv')
#     val_df = pd.read_csv('Plots/validation_data.csv')  # if available
#     induction_span_dist(save=False, exp_df=exp_df, val_df=val_df)
