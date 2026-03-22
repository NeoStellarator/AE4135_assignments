import pandas as pd
import matplotlib.pyplot as plt

def forces_span_dist(save=False, exp_df=None, val_df=None):
    """
    Plot axial and tangential force distributions along the span.
    
    Parameters:
    save (bool): If True, save the figure as PNG
    exp_df (DataFrame): Experimental/reference dataframe with 'r_R', 'Cx', 'Cy' columns
    val_df (DataFrame): Validation results dataframe with 'r_R', 'Cx', 'Cy' columns
    """
    
    # Plot experimental/reference data (solid lines)
    if exp_df is not None:
        plt.plot(exp_df['r_R'], exp_df['Cx'], '-', label='Axial force (Our model)', color='blue', linewidth=2)
        plt.plot(exp_df['r_R'], exp_df['Cy'], '--', label='Tangential force (Our model)', color='red', linewidth=2)
    
    # Plot validation data (dotted lines)
    if val_df is not None:
        plt.plot(val_df['r_R'], val_df['Cx'], ':', label='Axial force (JavaProp)', color='blue', linewidth=2, marker='o', markersize=4)
        plt.plot(val_df['r_R'], val_df['Cy'], ':', label='Tangential force (JavaProp)', color='red', linewidth=2, marker='s', markersize=4)
    
    plt.ylabel('Force coefficients')

def forces_span_dist(save=False, exp_df=None, val_df=None):
    """
    Plot axial and tangential force distributions along the span.
    
    Parameters:
    save (bool): If True, save the figure as PNG
    exp_df (DataFrame): Experimental/reference dataframe with 'r_R', 'Cx', 'Cy' columns
    val_df (DataFrame): Validation results dataframe with 'r_R', 'Cx', 'Cy' columns
    """
    
    # Plot experimental/reference data (solid lines)
    if exp_df is not None:
        plt.plot(exp_df['r_R'], exp_df['Cx'], '-', label='Axial force (Our model)', color='blue', linewidth=2)
        plt.plot(exp_df['r_R'], exp_df['Cy'], '--', label='Tangential force (Our model)', color='red', linewidth=2)
    
    # Plot validation data (dotted lines)
    if val_df is not None:
        plt.plot(val_df['r_R'], val_df['Cx'], ':', label='Axial force (JavaProp)', color='blue', linewidth=2, marker='o', markersize=4)
        plt.plot(val_df['r_R'], val_df['Cy'], ':', label='Tangential force (JavaProp)', color='red', linewidth=2, marker='s', markersize=4)
    
    plt.ylabel('Force coefficients')
    plt.xlabel('Spanwise location (r/R)')
    plt.title('Axial and azimuthal loading vs Spanwise location')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
    if save:
        plt.savefig('forces_span_dist.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Example usage
    # Load your dataframes here
    # exp_df = pd.read_csv('experimental_data.csv')
    # val_df = pd.read_csv('validation_results.csv')
    # forces_span_dist(save=False, exp_df=exp_df, val_df=val_df)
    
    # Or if you have only one dataframe:
    # forces_span_dist(save=False, exp_df=df)
    
    # Or if both dataframes are the same:
    # forces_span_dist(save=False, exp_df=df, val_df=df_validation)
    pass