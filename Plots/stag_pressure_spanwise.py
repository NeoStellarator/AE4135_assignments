import pandas as pd
import matplotlib.pyplot as plt
def stag_pressure_spanwise_loc(location):
    V0 = 60 #m/s
    rho = 1.007
    p0 = 7.950e4 #N/m^2
    df = pd.read_csv('Plots/BEM_randomdata.csv')
    if location == 'upwind':
        V = V0
        pstag = p0 + 0.5*rho*V**2
    elif location == 'rotor_up':
        V = V0*(1-df['a'])
        # stagnation pressure is constant before the rotor
        p_stag = p0 + 0.5*rho*V0**2
    elif location == 'rotor_down':
        #increase in pressure due to rotor
        V = V0*(1-df['a'])
        Vr = 1/2*(V0+V0*(1-2*df['a']))
        p_static_br = p0 + 0.5*rho*V0**2 - 0.5*rho*V**2
        p_jump = 2*rho*(V0-Vr)*Vr
        p_static_ar = p_static_br - p_jump #check signs should be a increase in pressure
        p_stag = p_static_ar + 0.5*rho*Vr**2
    else:
        V4 = V0*(1-2*df['a'])
        V2 = V0*(1-df['a'])
        Vr = 1/2*(V0+V4)
        p_static_br = p0 + 0.5*rho*V0**2 - 0.5*rho*V2**2
        p_jump = 2*rho*(V0-Vr)*Vr
        p_static_ar = p_static_br - p_jump #check signs should be a increase in pressure
        

    pstag = p0 + 0.5*rho*V**2
    
    