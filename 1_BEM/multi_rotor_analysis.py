from functools import partial

import numpy as np

from Rotor import Rotor
from globals import data_dir, res_dir

def run(option:str, 
        data:str,
        fname:str='result_Varying',
        polar_name:str='ARAD8pct_polar.txt') -> None:

    if '6' in polar_name:
        pitch_diff = -1 # we ran javaprop with a slightly different pitch!
    else:
        pitch_diff = 0

    PreLoadedRotor = partial( Rotor, 
        # Geometry definiton
        polar_path  = data_dir.joinpath(polar_name),
        c_R_func    = lambda r_R : 0.18-0.03*r_R,
        twst_func   = lambda r_R : -50*r_R+35,
        B           = 6,
        R           = 0.7,
        pitch       = 46 + pitch_diff,
        r_R_H       = 0.25,
        isPropeller = True,
        # Operating Condition
        Vinf        = 60,
        rho         = 1.067,
    )

    dist_opt = ['uniform','cosine']
    dist_elem = "uniform"

    n_elem    = 100
    n_lst = np.linspace(1,100,5,dtype=int)

    J         = 1.2
    J_lst = np.linspace(1.2,2.7,10)

    
    if option == 'J':
        for i,j in enumerate(J_lst):
            save_fpath = res_dir.joinpath(f'{fname}_J_{data}.csv')
            rotor = PreLoadedRotor(
                n_elem = n_elem,
                J  = j,
                dist_elem = dist_elem,
            )
            rotor.export_total(i+2,save_fpath)
    elif option == 'n':
        for i, n in enumerate(n_lst):
            save_fpath = res_dir.joinpath(f'{fname}_n_{data}.csv')
            rotor = PreLoadedRotor(
                n_elem = n,
                J  = J,
                dist_elem = dist_elem,
            )
            rotor.export_total(i+2,save_fpath)
    elif option == 'distribution':
        save_fpath = res_dir.joinpath(f'{fname}_distribution_{data}.csv')
        for i, dist in enumerate(dist_opt):
            rotor = PreLoadedRotor(
                n_elem = n_elem,
                J  = J,
                dist_elem = dist,
            )
            rotor.export_total(i+2,save_fpath)
    else:
        raise IndexError("Invalid option. Choose 'J', 'n', or 'distribution'.")

    return save_fpath



if __name__ ==  "__main__":
    run('distribution',"val2")
