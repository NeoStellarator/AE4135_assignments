from typing import Callable, List, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from Annuli import Annuli

from globals import data_dir



class Rotor:
    def __init__(self,Uinf, Omega, c_R_func,twst_func, B, R, pitch, r_R_H,n_elem,polar_path,H=2000,dist_elem='uniform',isPropeller = False):
        self.Uinf = Uinf
        self.Omega = Omega
        self.c_R_func = c_R_func
        self.twst_func = twst_func
        self.B = B
        self.R = R
    def __init__(
        self,
        # Geometry
        B : int,
        R : float,
        r_R_H : float,
        c_R_func : Callable[[float|np.ndarray], float|np.ndarray],
        twst_func : Callable[[float|np.ndarray], float|np.ndarray],
        pitch : float,
        polar_path : Path|str,
        # Operating condition
        J : float,
        Vinf : float,
        rho : float,
        # Other parameters
        n_elem:int=100,
        dist_elem:Literal['uniform','cosine']='uniform',
        isPropeller:bool=True
    ):
        # Rotor geometry
        self.B = B                    
        self.R    = R
        self.r_R_H = r_R_H            
        self.c_R_func = c_R_func      
        self.twst_func = twst_func    
        self.pitch = pitch
        self.r_R_H = r_R_H
        self.n_elem = n_elem
        self.polar_path = polar_path
        self.dist_elem = dist_elem
        self.isPropeller = isPropeller

        # Operating Conditions
        self.J    = J                    
        self.Vinf = Vinf
        self.n    = self.Vinf/(self.J*2*self.R)
        self.Ome  = self.n*2*np.pi
        self.rho  = rho

        # Discretizaiton scheme
        self.n_elem = n_elem
        self.dist_elem = dist_elem
        
        if dist_elem == "uniform":
            rr = np.linspace(r_R_H, 1, n_elem+1)
            self.r1_R_lst = rr[:-1]
            self.r2_R_lst = rr[1:]
            
            self.r_R_lst = (rr[1:] + rr[:-1])/2
            
            self.c_R_lst = c_R_func(self.r_R_lst)
            self.beta_lst = pitch + twst_func(self.r_R_lst)
        self.T = 288.15 - 0.0065 * H
        self.rho = 1.225* (self.T/288.15)**(-(9.81/(-0.0065*287)+1))
        print(self.rho)
        self.annuli_lst = [
            
            Annuli(polar_path=polar_path,
                   Uinf = Uinf,
                   Omega= Omega,
                 r1_R=self.r1_R_lst[i],
                 r2_R=self.r2_R_lst[i],
                 r_R_H=self.r_R_H,
                 R=self.R,
                 B=self.B,
                 c_R=self.c_R_lst[i],
                 beta=self.beta_lst[i],
                 rho = self.rho,
                 isPropeller = self.isPropeller,
                 a0=1/3,
                 aline0=0) for i in range(n_elem)]
    def plot_radial(self):
        self.phi_lst   = [an.phi   for an in self.annuli_lst]
        self.alpha_lst = [an.alpha for an in self.annuli_lst]
        self.Cl_lst = [an.Cl for an in self.annuli_lst]
        self.Cd_lst = [an.Cd for an in self.annuli_lst]
        self.Cx_lst = [an.Cx for an in self.annuli_lst]
        self.Fy_lst = [an.Cy for an in self.annuli_lst]
        self.f_lst  = [an.f  for an in self.annuli_lst]
        self.CT_lst = [an.Ct for an in self.annuli_lst]
        self.a_lst  = [an.a  for an in self.annuli_lst]
        self.aline_lst = [an.aline for an in self.annuli_lst]
        self.Ux_lst = [an.Ux for an in self.annuli_lst]
        self.Uy_lst = [an.Uy for an in self.annuli_lst]

        fig, axs = plt.subplots(3, 2)

        # (1) phi & alpha
        ax = axs[0]
        ax.plot(self.r_R, self.phi, label=r'$\phi$')
        ax.plot(self.r_R, self.alpha, label=r'$\alpha$')
        ax.set_title("Angles")
        ax.legend()
        ax.grid()

        # (2) Cl, Cd, CT
        ax = axs[1]
        ax.plot(self.r_R, self.Cl, label=r'$C_l$')
        ax.plot(self.r_R, self.Cd, label=r'$C_d$')
        ax.plot(self.r_R, self.Ct, label=r'$C_T$')
        ax.set_title("Coefficients")
        ax.legend()
        ax.grid()

        # (3) f
        ax = axs[2]
        ax.plot(self.r_R, self.f, label='f')
        ax.set_title("f")
        ax.legend()
        ax.grid()

        # (4) a & aline
        ax = axs[3]
        ax.plot(self.r_R, self.a, label='a')
        ax.plot(self.r_R, self.aline, label="a'")
        ax.set_title("Induction Factors")
        ax.legend()
        ax.grid()

        ax = axs[2, 0]
        ax.plot(self.r_R_lst, self.Ux_lst, label=r'$U_x$')
        ax.plot(self.r_R_lst, self.Uy_lst, label=r'$U_y$')
        ax.set_title("Velocities")
        ax.legend()
        ax.grid()

        plt.tight_layout()
        plt.show()
    def calculate_integral(self):
        integral_values = np.array([an.calculate_forces() for an in self.annuli_lst])
        print(integral_values)
        total_thrust = sum(integral_values[:,0])
        total_torque = sum(integral_values[:,1])
        n= self.Omega/(2*np.pi)*60
        total_CT = total_thrust/(self.rho*n**2*(2*self.R)**4)
        return total_thrust, total_torque, total_CT
    def print_geometry(self):
        for i in range(self.n_elem):
            print(self.r_R_lst[i],self.c_R_lst[i],self.beta_lst[i])

        

if __name__ == "__main__":
    # Propeller
    c_R_func:Callable = lambda r_R : 0.18-0.06*r_R
    twst_func:Callable = lambda r_R : -50*r_R+35+60
    B:float=6
    R:float=0.7
    pitch:float=46
    r_R_H:float=0.25
    n_elem:int=100
    polar:str|Path=main_dir.joinpath('ARAD8pct_polar.txt')
    polar = 'ARAD8pct_polar.txt'
    pitch_ref:float=0
    dist_elem:str="uniform"
    Uinf = 60
    n=1200
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

    # # WIND TURBINE
    # R:float=50
    # c_R_func:Callable = lambda r_R : (3*(1-r_R)+1)/R
    # twst_func:Callable = lambda r_R : 14*(1-r_R)
    # B:float=3
    # J:float=np.pi/8
    # pitch:float=-2
    # r_R_H:float=0.20
    # n_elem:int=1000
    # polar='DU95W180.txt'
    # pitch_ref:float=0
    # dist_elem:str="uniform"
    # Uinf = 10
    # TSR = 8
    # Omega = TSR*Uinf/R

    # rotor = Rotor(
    #     Uinf=Uinf,
    #     Omega=Omega,
    #     c_R_func=c_R_func, 
    #     twst_func=twst_func,
    #     B=B,
    #     R=R,
    #     pitch=pitch,
    #     r_R_H=r_R_H,
    #     n_elem=n_elem,
    #     polar_path=polar,
    #     isPropeller=False)
    
<<<<<<< HEAD
    
    rotor.plot_radial()
=======
    # rotor.print_geometry()
    print(rotor.calculate_integral())
    rotor.plot_radial()
    
>>>>>>> 69efa88 (added thrust and torque calculation)
