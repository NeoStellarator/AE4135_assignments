from Rotor import Rotor
from pathlib import Path
from typing import Callable
import numpy as np


from globals import main_dir
def run_rotor_diff_inputs(option):
    c_R_func:Callable = lambda r_R : 0.18-0.06*r_R
    twst_func:Callable = lambda r_R : -50*r_R+35
    B:float=6
    R:float=0.7
    pitch:float=46
    r_R_H:float=0.25
    n_elem:int=100
    polar:str|Path=main_dir.joinpath('ARAD8pct_polar.txt')
    polar = '1_BEM/ARAD8pct_polar.txt'
    pitch_ref:float=0
    dist_elem:str="uniform"
    Uinf = 60
    n=1200/60
    Omega = n*2*np.pi/60
    if option == 'J':
        J_lst = np.linspace(1.2,2.7,10)
        for i,j in enumerate(J_lst):
            n=Uinf/(j*2*R)
            Omega = n*2*np.pi/60
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

            rotor.write_Total_res_for_Input(i+2,'results_Varying_J.csv')
    elif option == 'n':
        n_lst = np.linspace(5,300,20)
        for i,n in enumerate(n_lst):
            rotor = Rotor(
                Uinf=Uinf,
                Omega=Omega,
                c_R_func=c_R_func, 
                twst_func=twst_func,
                B=B,
                R=R,
                pitch=pitch,
                r_R_H=r_R_H,
                n_elem=n,
                polar_path=polar,
                isPropeller=True)
            rotor.write_Total_res_for_Input(i+2,'results_Varying_n.csv')
    elif option == 'distribution':
        dist_opt = ['uniform','cosine']
        for i,dist in enumerate(dist_opt):
            rotor = Rotor(
                Uinf=Uinf,
                Omega=Omega,
                c_R_func=c_R_func, 
                twst_func=twst_func,
                B=B,
                R=R,
                pitch=pitch,
                r_R_H=r_R_H,
                n_elem=n,
                polar_path=polar,
                dist_elem=dist,
                isPropeller=True)
            rotor.write_Total_res_for_Input(i+2,'results_Varying_distribution.csv')
    else:
        print("Invalid option. Choose 'J', 'n', or 'distribution'.")

run_rotor_diff_inputs('distribution')
