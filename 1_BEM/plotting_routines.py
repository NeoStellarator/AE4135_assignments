from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


from globals import plot_dir

label_dict = {
    'a': r'$a$ [-]',
    'aline' : r"$a'$ [-]",
    'alpha' : r"$\alpha$ [deg]",
    'phi' : r"$\phi$ [deg]",
    'Cx' : r'$C_n$ [-]',
    'Cy' : r'$C_t$ [-]',
    'T' : r"$T$ [N]",
    'Q' : r"$Q$ [Nm]",
    'J' : r"$J$ [-]",  
    'Ct' : r"$C_T$ [-]"

}


def forces(plot_vs: Literal['J', 'n_elem'],
           plot_ag: Literal['T', 'Q'],
           data_df:pd.DataFrame,
           ax:Axes=None,
           verif:bool=False,
           **kwargs) -> Axes:

    x = data_df[plot_vs]

    if verif:
        y = data_df['Thrust' if plot_ag=='T' else 'Torque'] * 1000
    else:
        y = data_df[plot_ag]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))    
    
    ax.plot(x, y, **kwargs)

    if plot_vs == 'J':
        ax.set_xlabel(r"$J$ [-]")
    elif plot_vs == 'n_elem':
        ax.set_xlabel(r"# Elements [-]")
    else:
        raise ValueError("Plot_vs not recognised!")
    
    ax.set_ylabel(label_dict[plot_ag])
    ax.grid(True,alpha=0.3)

    return ax


def distribution(plot_vs: Literal['a', 'aline', 'alpha', 'phi', 'Cx', 'Cy','Ct'],
                 data_df:pd.DataFrame,
                 ax:Axes=None,
                 **kwargs) -> Axes:

    x = data_df['r_R']
    y = data_df[plot_vs]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))    
    
    ax.plot(x, y, **kwargs)

    ax.set_xlabel(r"$r/R$ [-]")
    ax.set_ylabel(label_dict[plot_vs])
    ax.grid(True, alpha=0.3)

    return ax



def barchart_forces_vs_dist(save=False, csv_path=None):
    """
    Create a bar chart comparing total thrust and torque for Uniform and Cosine distributions.
    
    Parameters:
    save (bool): If True, save the figure as PNG
    csv_path (str): Path to the CSV file containing the data
    """
    
    # Load the data if csv_path is provided
    df = pd.read_csv(csv_path)
    
    # Define the two distributions
    distributions = ['Uniform', 'Cosine']
    
    # Extract the data for each distribution
    thrust_uniform = df[df.index == 0]['Total Thrust'].values[0]
    torque_uniform = df[df.index == 0]['Torque'].values[0]
    thrust_cosine = df[df.index == 1]['Total Thrust'].values[0]
    torque_cosine = df[df.index == 1]['Torque'].values[0]
    
    # Create the bar chart
    x = np.arange(len(distributions))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars for Thrust and Torque
    rects1 = ax.bar(x - width/2, [thrust_uniform, thrust_cosine], width, 
                    label='Total Thrust', color='blue', alpha=0.8)
    rects2 = ax.bar(x + width/2, [torque_uniform, torque_cosine], width, 
                    label='Torque', color='red', alpha=0.8)
    
    # Add labels, title, and legend
    ax.set_ylabel('Forces [N and N-m]', fontsize=12)
    ax.set_xlabel('Distribution Configuration', fontsize=12)
    ax.set_title('Total Thrust and Torque Comparison: Uniform vs Cosine Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(distributions)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    def autolabel(rects):
        """Attach a text label above each bar displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Adjust layout and display/save
    plt.tight_layout()
    
    if save:
        plt.savefig(plot_dir.joinpath('forces_comparison.pdf'), bbox_inches='tight')
        print(f"Figure saved as 'forces_comparison.pdf'")
    
    plt.show()

def convergence_history(data_df:pd.DataFrame,
           ax:Axes=None,
           mark:bool = True,
           dist:str=None):
    
    iter = []
    total_thrust_lst = []
    df = data_df.ffill(axis=1)
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
    
    
    ax.set_label(['Iteration #','Total Thrust [N]'])
    ax.grid(True, alpha=0.3)
    #Mark the final point
    if mark:
        ax.plot(iter, total_thrust_lst, marker='o')
        ax.plot(iter[-1], total_thrust_lst[-1], marker='o', markersize=10, color='red', label={f'Final thrust: {total_thrust_lst[-1]:.2f} N'})
    else:
        label = {f'{dist} with final thrust: {total_thrust_lst[-1]:.2f} N'}
        ax.plot(iter, total_thrust_lst, marker='o',label=label)
    ax.legend()
