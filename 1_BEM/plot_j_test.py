from Rotor import Rotor
from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt
c_R_func:Callable = lambda r_R : 0.18-0.06*r_R
twst_func:Callable = lambda r_R : -50*r_R+35
B:float=6
R:float=0.7
pitch:float=46
r_R_H:float=0.25
n_elem:int=100
# polar:str|Path=main_dir.joinpath('ARAD8pct_polar.txt')
polar = 'ARAD8pct_polar.txt'
pitch_ref:float=0
dist_elem:str="uniform"
Uinf = 60
n=1200
Omega = n*2*np.pi/60

Uinf_lst= np.linspace(5,80,50)
CT_lst= []
J_lst = []
efficiency_lst = []
for Uinf in Uinf_lst:
    rotor = Rotor(
        Uinf=Uinf,
        Omega=Omega,
        c_R_func=c_R_func, 
        twst_func=twst_func,
        B=B,
        R=R,
        pitch=pitch,
        r_R_H=r_R_H,
        n_elem=n_elem,
        polar_path=polar,
        isPropeller=True)
    total_thrust, total_torque, total_power, total_CT, total_CQ, total_CP,propeller_efficiency = rotor.calculate_integral()
    CT_lst.append(total_CT)
    J_lst.append(rotor.J)
    efficiency_lst.append(propeller_efficiency)
    if rotor.J>2.6:
        break

# plt.figure()
# plt.plot(J_lst,CT_lst)
# plt.xlabel("J")
# plt.ylabel(r"$C_T$")
# plt.show()

plt.figure()
plt.plot(J_lst,efficiency_lst)
plt.xlabel("J")
plt.ylabel(r"$\eta$")
plt.show()

