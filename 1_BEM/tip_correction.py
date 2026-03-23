import numpy as np
import matplotlib.pyplot as plt

def ning_correction(B:float,
                    phi:float,
                    r_R:float,
                    r_R_H:float=0)-> float:
    """Glauert tip-correction based on inflow angle, applied on forces.
    
    Source: Ning 
    """
    R_r = 1/r_R
    f_tip = B/2 * (R_r-1)/abs(np.sin(phi))
    F_tip = 2/np.pi * np.arccos(np.exp(-f_tip))

    f_hub = B/2 * (r_R/r_R_H-1)/abs(np.sin(phi))
    F_hub = 2/np.pi * np.arccos(np.exp(-f_hub))
    
    return max(F_tip*F_hub, 1e-6)
    

def prandtl_correction(B:float,
                       phi:float,
                       r_R:float,
                       r_R_H:float=0)-> float:
    """
    Source: 2.84 - 'exact'correction suugested by Betz and Prandtl, assuming
    Vn = Uinf(1-a) and Vt = Ome*r*(1+a') [E. Branlard 2011]

    modified in terms of phi
    """
    d = -B/2*1/np.abs(np.sin(phi))
    f_tip  = 2/np.pi*np.arccos(np.exp(d*(1-r_R)))
    f_root = 2/np.pi*np.arccos(np.exp(d*(r_R-r_R_H)))
    return max(f_tip*f_root, 1e-6)


if __name__ == "__main__":
    B = 40
    phi = np.deg2rad(40)

    r_R = np.linspace(0,1,100)
    f1 = [ning_correction(B, phi, r_R_i, 0.25) for r_R_i in r_R]
    f2 = [prandtl_correction(B, phi, r_R_i, 0.25) for r_R_i in r_R]

    fig, ax = plt.subplots()

    ax.plot(r_R, f1, label='ning')
    ax.plot(r_R, f2, label='prandtl')

    ax.legend()

    plt.show()