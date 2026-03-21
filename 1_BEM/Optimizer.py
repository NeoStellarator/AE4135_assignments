import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from Rotor import Rotor

from globals import main_dir

def normalize(x,bounds):
    x_norm = [(x[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(x))]
    bounds_norm = [[0,1] for i in range(len(x))]
    return x_norm,bounds_norm
def denormalize(x_norm, bounds):
    x = [
        bounds[i][0] + x_norm[i] * (bounds[i][1] - bounds[i][0])
        for i in range(len(x_norm))
    ]
    return x
def objective(x_norm, args):
    B, J, r_R_H, polar_path, n_elem, dist_elem, isPropeller, R, Uinf, rho, bounds = args
    x = denormalize(x_norm,bounds)
    at = x[0]
    bt = x[1]
    ac = x[2]
    bc = x[3]
    
    twst_func = lambda r_R : at*r_R+bt
    c_R_func = lambda r_R : bc+ac*r_R
    try:
        rotor = Rotor(c_R_func=c_R_func,
                    twst_func=twst_func,
                    pitch=0,
                    B=B,
                    J=J,
                    r_R_H=r_R_H,
                    polar_path=polar_path,
                    n_elem=n_elem,
                    dist_elem=dist_elem,
                    isPropeller=isPropeller)
        thrust, azimuthal, torque, power=rotor.calculate_integral(R=R,Uinf=Uinf,rho=rho)
    except Exception as e:
        power = 1E6
        print(e)
    return power

bounds = [[-90,90],
          [10,120],
          [-0.02,-0.10],
          [0.10,0.30]]
x0 = [-50,35+46,-0.06,0.18]

x0_norm, bounds_norm = normalize(x0,bounds)
B:float=6
R:float=0.7
r_R_H:float=0.25
n_elem:int=100
polar_path:str|Path=main_dir.joinpath('ARAD8pct_polar.txt')
dist_elem:str="uniform"
isPropeller =True
Uinf = 60
rho = 1
J=60/20/1.4
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
x = denormalize(x_norm,bounds)
at = x[0]
bt = x[1]
ac = x[2]
bc = x[3]
twst_func = lambda r_R : at*r_R+bt
c_R_func = lambda r_R : bc+ac*r_R
rotor = Rotor(
        c_R_func=c_R_func,
        twst_func=twst_func,
        pitch=0,
        B=B,
        J=J,
        r_R_H=r_R_H,
        n_elem=n_elem,
        polar_path=main_dir.joinpath('ARAD8pct_polar.txt'),
        dist_elem='uniform',
        isPropeller=True)
rotor.plot_radial()