import numpy as np
import matplotlib.pyplot as plt


def calculate_prandtl_correction(B:float,
                                 TSR:float,
                                 a:float,
                                 r_R:float,
                                 a_:float=0,
                                 r_R_H:float=0)->float:
    """Method to compute prandtl hub/tip-loss correction"""

    d = (2*np.pi/B)*(1-a)/np.sqrt(TSR**2+(1-a)**2)
    
    f_tip  = 2/np.pi*np.arccos(np.exp((-np.pi*(1-r_R)/d)))
    f_root = 2/np.pi*np.arccos(np.exp((-np.pi*(r_R)/d))) #reversed
    
    return f_tip*f_root


def calculate_prandtl_correction2(B:float,
                                 TSR:float,
                                 a:float,
                                 a_line:float,
                                 r_R:float,
                                 r_R_H:float=0)-> float:
    """
    Source: 2.84 - 'exact'correction suugested by Betz and Prandtl, assuming
    Vn = Uinf(1-a) and Vt = Ome*r*(1+a') [E. Branlard 2011]

    """
    d = -B/2*np.sqrt(1+TSR**2*((1+a_line)/(1-a))**2)
    f_tip  = 2/np.pi*np.arccos(np.exp(d*(1-r_R)))
    f_root = 2/np.pi*np.arccos(np.exp(d*(r_R-r_R_H)))
    return f_tip*f_root



if __name__ == "__main__":
    B = 40
    TSR = 1.4
    a = 0.001
    a_line = 0.0001

    r_R = np.linspace(0,1,100)
    f1 = calculate_prandtl_correction(B, TSR, a, r_R)
    f2 = calculate_prandtl_correction2(B, TSR, a, a_line, r_R)

    fig, ax = plt.subplots()

    ax.plot(r_R, f1, label='og')
    ax.plot(r_R, f2, label='v2')

    ax.legend()

    plt.show()