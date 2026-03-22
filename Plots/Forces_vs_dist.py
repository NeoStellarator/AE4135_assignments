import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def forces_vs_dist(save=False, csv_path=None):
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
        plt.savefig('forces_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved as 'forces_comparison.png'")
    
    plt.show()
    


if __name__ == "__main__":
    # Example usage with CSV file
    # forces_vs_n(save=True, csv_path='your_data.csv')
    
    # Or use the embedded sample data
    forces_vs_n(save=False)