import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from Rotor import Rotor

from globals import data_dir

def normalize(x, bounds):
    x_norm = [(x[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(x))]
    bounds_norm = [[0,1] for i in range(len(x))]
    return x_norm, bounds_norm

def denormalize(x_norm, bounds):
    x = [
        bounds[i][0] + x_norm[i] * (bounds[i][1] - bounds[i][0])
        for i in range(len(x_norm))
    ]
    return x

def objective(x_norm, args):
    B, J, r_R_H, polar_path, n_elem, dist_elem, isPropeller, R, Uinf, rho, bounds = args
    x = denormalize(x_norm, bounds)
    at = x[0]
    bt = x[1]
    ac = x[2]
    bc = x[3]
    
    twst_func = lambda r_R: at*r_R + bt
    c_R_func = lambda r_R: bc + ac*r_R
    try:
        # Updated Rotor initialization for new class
        rotor = Rotor(
            c_R_func=c_R_func,
            twst_func=twst_func,
            pitch=0,
            B=B,
            R=R,                    # Added R parameter
            r_R_H=r_R_H,
            polar_path=polar_path,
            J=J,
            Vinf=Uinf,              # Changed from Uinf to Vinf
            rho=rho,
            n_elem=n_elem,
            dist_elem=dist_elem,
            isPropeller=isPropeller
        )
        
        # Get power directly from rotor attribute
        power = rotor.P
        
    except Exception as e:
        power = 1E6
        print(e)
    return power

bounds = [[-90, 90],
          [10, 120],
          [-0.10, -0.02],   # Fixed order (min, max)
          [0.10, 0.30]]
x0 = [-50, 35+46, -0.06, 0.18]

x0_norm, bounds_norm = normalize(x0, bounds)

B: float = 6
R: float = 0.7
r_R_H: float = 0.25
n_elem: int = 100
polar_path: str | Path = data_dir.joinpath('ARAD8pct_polar.txt')
dist_elem: str = "uniform"
isPropeller = True
Uinf = 60
rho = 1
J = 60/20/1.4

arguments = [B, J, r_R_H, polar_path, n_elem, dist_elem, isPropeller, R, Uinf, rho, bounds]

res = minimize(
    fun=objective,
    args=arguments,
    x0=x0_norm,
    method="L-BFGS-B",
    bounds=bounds_norm
)

print(res)

x_norm = res.x
x = denormalize(x_norm, bounds)
at = x[0]
bt = x[1]
ac = x[2]
bc = x[3]

twst_func = lambda r_R: at*r_R + bt
c_R_func = lambda r_R: bc + ac*r_R

# Updated Rotor initialization for final evaluation
rotor = Rotor(
    c_R_func=c_R_func,
    twst_func=twst_func,
    pitch=0,
    B=B,
    R=R,                        # Added R parameter
    r_R_H=r_R_H,
    n_elem=n_elem,
    polar_path=data_dir.joinpath('ARAD8pct_polar.txt'),
    dist_elem='uniform',
    J=J,                        # Added J parameter
    Vinf=Uinf,                  # Changed from Uinf to Vinf
    rho=rho,
    isPropeller=isPropeller
)

# Get results directly from rotor attributes
final_power = rotor.P
print(f"\nFinal power: {final_power:.2f} W")
print(f"Optimized parameters: at={at:.2f}, bt={bt:.2f}, ac={ac:.4f}, bc={bc:.4f}")

# Plot results (using plot_check instead of plot_radial)
rotor.plot_check()