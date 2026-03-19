import numpy as np
import matplotlib.pyplot as plt


def calculate_prandtl_correction(B:float,
                                 TSR:float,
                                 a:float,
                                 r_R:float,
                                 a_line,
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
import numpy as np

def calculate_prandtl_correction_3(
    B: float,
    TSR: float,
    a: float,
    a_line: float,
    r_R: float,
    r_R_H: float = 0.0
):
    """
    Compute Prandtl tip and root loss correction factor.

    Parameters
    ----------
    B : float
        Number of blades
    TSR : float
        Tip speed ratio (λ)
    a : float
        Axial induction factor
    a_line : float
        Tangential induction factor (a')
    r_R : float
        Radial position (r/R)
    r_R_H : float, optional
        Hub radius ratio (R_hub / R), default = 0.0

    Returns
    -------
    F : float
        Combined Prandtl correction factor
    F_tip : float
        Tip loss factor
    F_root : float
        Root loss factor
    """

    # --- Numerical safety ---
    eps = 1e-8
    r_R = np.clip(r_R, r_R_H + eps, 1.0 - eps)

    # --- Flow angle term ---
    denom = max(1.0 + a, eps)
    phi_term = np.sqrt(1.0 + (TSR * r_R / denom) ** 2)

    # --- Exponential arguments ---
    exponent_tip = - (B / 2.0) * ((1.0 - r_R) / r_R) * phi_term
    exponent_root = - (B / 2.0) * ((r_R - r_R_H) / r_R) * phi_term

    # --- Prevent overflow in exp ---
    exp_tip = np.exp(np.clip(exponent_tip, -700.0, 0.0))
    exp_root = np.exp(np.clip(exponent_root, -700.0, 0.0))

    # --- Clamp for arccos domain ---
    exp_tip = np.clip(exp_tip, 0.0, 1.0)
    exp_root = np.clip(exp_root, 0.0, 1.0)

    # --- Prandtl factors ---
    F_tip = (2.0 / np.pi) * np.arccos(exp_tip)
    F_root = (2.0 / np.pi) * np.arccos(exp_root)

    # --- Combined correction ---
    F = F_tip * F_root
    F = np.clip(F, 1e-4, 1.0)

    return F
    



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