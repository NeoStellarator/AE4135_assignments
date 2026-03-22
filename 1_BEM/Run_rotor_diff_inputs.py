from Rotor import Rotor
from pathlib import Path
from typing import Callable
import numpy as np

from globals import data_dir
def run_rotor_diff_inputs(option,data):
    c_R_func:Callable = lambda r_R : 0.18-0.03*r_R
    twst_func:Callable = lambda r_R : -50*r_R+35
    B:float=6
    R:float=0.7
    pitch:float=46
    r_R_H:float=0.25
    n_elem:int=100
    polar_path = data_dir.joinpath('ARAD6pct_polarOWN.txt'),
    dist_elem:str="uniform"
    J=60/20/1.4

    
    if option == 'J':
        J_lst = np.linspace(1.2,2.7,10)
        for i,j in enumerate(J_lst):
            rotor = Rotor(
        # Geometry definition
        c_R_func  = c_R_func,
        twst_func = twst_func,
        pitch = 45,
        B     = 6,
        R     = 0.7,
        r_R_H = 0.25,
        polar_path = data_dir.joinpath('ARAD6pct_polarOWN.txt'),
        # Operating condition
        J = j,
        Vinf = 60,
        rho = 1.067,
        # Discretizaiton
        n_elem = 100,
        dist_elem = 'uniform',
        isPropeller = True)

            rotor.write_Total_res_for_Input(i+2,f'results_Varying_J_{data}.csv')
    elif option == 'n':
        n_lst = np.linspace(5,300,20,dtype=int)
        print(n_lst)
        for i,n in enumerate(n_lst):
            rotor = Rotor(
            # Geometry definition
            c_R_func  = c_R_func,
            twst_func = twst_func,
            pitch = 45,
            B     = 6,
            R     = 0.7,
            r_R_H = 0.25,
            polar_path = data_dir.joinpath('ARAD8pct_polar.txt'),
            # Operating condition
            J = 1.2,
            Vinf = 60,
            rho = 1.067,
            # Discretizaiton
            n_elem = n,
            dist_elem = 'uniform',
            isPropeller = True)
            rotor.write_Total_res_for_Input(i+2,f'results_Varying_n_{data}.csv')
    elif option == 'distribution':
        dist_opt = ['uniform','cosine']
        for i,dist in enumerate(dist_opt):
            rotor = Rotor(
        # Geometry definition
        c_R_func  = c_R_func,
        twst_func = twst_func,
        pitch = 45,
        B     = 6,
        R     = 0.7,
        r_R_H = 0.25,
        polar_path = data_dir.joinpath('ARAD8pct_polar.txt'),
        # Operating condition
        J = 1.2,
        Vinf = 60,
        rho = 1.067,
        # Discretizaiton
        n_elem = 100,
        dist_elem = dist,
        isPropeller = True)
            rotor.write_Total_res_for_Input(i+2,f'results_Varying_distribution_{data}.csv')
    else:
        print("Invalid option. Choose 'J', 'n', or 'distribution'.")

run_rotor_diff_inputs('J',"val2")
