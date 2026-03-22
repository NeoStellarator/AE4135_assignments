import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def stag_pressure_spanwise_loc(locations,save=False):
    V0 = 60 #m/s
    rho = 1.007
    p0 = 7.950e4 #N/m^2
    df = pd.read_csv('Plots\propeller_radial_data.csv')
    def Bernulli_forP(p0, rho, V0,V1):
        return p0 + 0.5*rho*V0**2-0.5*rho*V1**2
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten to easily index
    
    for i, loc in enumerate(locations):
        if loc == 'far upwind' or loc == 'upwind of the rotor disk':
            V = V0
            p_stag = p0 + 0.5*rho*V**2
            p_stag = np.ones(len(df['r_R']))*p_stag
        else:
            #increase in pressure due to rotor
            V = V0*(1-df['a'])
            Vr = 1/2*(V0+V0*(1-2*df['a']))
            p_static_br = Bernulli_forP(p0,rho,V0,V)
            p_jump = 2*rho*(V0-Vr)*Vr
            p_static_ar = p_static_br - p_jump #check signs should be a increase in pressure
            if loc == 'downwind of the rotor disk':
                p_stag = p_static_ar + 0.5*rho*V**2
            elif loc == 'far downwind':
                pstatic = Bernulli_forP(p_static_ar,rho,V,Vr)
                p_stag = pstatic + 0.5*rho*Vr**2
        
        # Plot on the appropriate subplot
        axes[i].plot(df['r_R'], p_stag, label=f'stagnation pressure at {loc}')
        axes[i].set_xlabel('r/R')
        axes[i].set_ylabel('Stagnation Pressure (N/m²)')
        axes[i].set_title(f'Stagnation Pressure at {loc}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig('stag_pressure_spanwise.png')
    plt.show()
    
locations = ['far upwind', 'upwind of the rotor disk', 'downwind of the rotor disk', 'far downwind']
stag_pressure_spanwise_loc(locations, save=True)
    
    