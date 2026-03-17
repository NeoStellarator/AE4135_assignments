from typing import Callable, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from Annuli import Annuli

from globals import main_dir
# def evaluate_rotor(
#     c_R_func:Callable,
#     twst_func:Callable,
#     B:float,
#     J:float,
#     pitch:float,
#     r_R_H:float,
#     n_elem:int,
#     polar:str|Path,
#     pitch_ref:float=0,
#     dist_elem:str="uniform"
# ):

# PROPELLER
c_R_func:Callable = lambda r_R : 0.18-0.06*r_R
twst_func:Callable = lambda r_R : -50*r_R+35
B:float=6
J:float=2
R:float=0.7
pitch:float=46
r_R_H:float=0.25
n_elem:int=100
polar:str|Path=main_dir.joinpath('ARAD8pct_polar.txt')
polar = 'ARAD8pct_polar.txt'
pitch_ref:float=0
dist_elem:str="uniform"

# WIND TURBINE
# R:float=50
# c_R_func:Callable = lambda r_R : (3*(1-r_R)+1)/R
# twst_func:Callable = lambda r_R : 14*(1-r_R)
# B:float=3
# J:float=np.pi/8
# pitch:float=-2
# r_R_H:float=0.20
# n_elem:int=100
# polar='DU95W180.txt'
# pitch_ref:float=0
# dist_elem:str="uniform"

# discretize blade
if dist_elem == "uniform":
    rr = np.linspace(r_R_H, 1, n_elem+1)

    dr_R_lst = rr[1:] - rr[:-1]
    r_R_lst  = rr[:-1] + dr_R_lst/2

else:
    NotImplementedError("Distribution unknown!")

# evaluate chord and twist
c_R_lst = c_R_func(r_R_lst)
beta_lst = pitch + twst_func(r_R_lst)


annuli = [
    Annuli(
        polar_path=polar,
        r_R=r_R_lst[i],
        c_R=c_R_lst[i],
        dr_R=dr_R_lst[i],
        beta=beta_lst[i],
        B=B,
        J=J,
        R=R,
        r_R_H=r_R_H
    )
    
    for i in range(n_elem)
]

# fetch relevant values
a_lst     = [an.a     for an in annuli]
aline_lst = [an.aline for an in annuli]
Cx_lst    = [an.Cx    for an in annuli]
Cy_lst    = [an.Cy    for an in annuli]
Cl_lst    = [an.Cl    for an in annuli]
Cd_lst    = [an.Cd    for an in annuli]
Ct_lst    = [an.Ct    for an in annuli]
Ca_lst    = [an.Ca    for an in annuli]
Cp_lst    = [an.Cq    for an in annuli]
Cq_lst    = [an.Cq    for an in annuli]
f_lst     = [an.f     for an in annuli]
phi_lst   = [an.phi   for an in annuli]
alpha_lst = [an.alpha for an in annuli]

fig, axs = plt.subplots(2,2)

ax = axs[0,0]
ax.plot(r_R_lst, a_lst, label="a")
ax.plot(r_R_lst, aline_lst, label="a'")
ax.legend()
ax.grid()


ax = axs[0,1]
ax.plot(r_R_lst, Cx_lst, label=r'$C_x$')
ax.plot(r_R_lst, Cy_lst, label=r'$C_y$')
ax.plot(r_R_lst, Ca_lst, label=r'$C_a$')
ax.plot(r_R_lst, Ct_lst, label=r'$C_t$')
ax.plot(r_R_lst, Cl_lst, label=r'$C_l$')
ax.plot(r_R_lst, Cd_lst, label=r'$C_d$')
ax.plot(r_R_lst, Cp_lst, label=r'$C_p$')
ax.plot(r_R_lst, Cq_lst, label=r'$C_q$')
ax.legend()
ax.grid()

ax = axs[1,1]
ax.plot(r_R_lst, phi_lst, label=r'$\phi$')
ax.plot(r_R_lst, alpha_lst, label=r'$\alpha$')
ax.legend()
ax.grid()


ax = axs[1,0]
ax.plot(r_R_lst, f_lst, label=r'f')
ax.legend()
ax.grid()

plt.show()