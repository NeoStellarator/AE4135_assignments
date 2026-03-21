from typing import Dict, List, Callable, Tuple
import numpy as np
from pathlib import Path

from globals import main_dir
import tip_correction
from scipy import optimize
import matplotlib.pyplot as plt

class Annuli:

    def __init__(self, 
                 polar_path:Path|str, 
                 r_R:float,
                 c_R:float,
                 beta:float,
                 B:int, 
                 J:float,
                 r_R_H:float=0,
                 is_prop:bool=True):

        # save blade information
        self.r_R   = r_R   # blade radial coordinate
        self.r_R_H = r_R_H # hub radial coordinate
        self.c_R   = c_R   # blade chord 
        self.beta  = beta  # blade twist
        self.B     = B     # blade number
        self.J     = J     # advance ratio
        self.TSR   = np.pi/J # tip speed ratio
        self.sig   = self.B/(2*np.pi)*self.c_R/self.r_R # blade solidity

        self.is_prop = is_prop # propeller or turbine option

        self.hist:Dict[str,List[float]] = dict()
        
        # read & store polar data
        self.polar_data = self._load_polar_data(polar_path)

        # compute bounds based on available data
        a_rng = np.array([self.polar_data["alpha"].min(), 
                          self.polar_data["alpha"].max()])
        self.phi_rng = np.deg2rad(self.beta - a_rng)

        print(np.rad2deg(self.phi_rng))
        # solve annuli
        self.solve()


    def _update_hist(self, snap:Dict[str,str])-> None:
        """Method to update history"""
        for k,v in snap.items():
            if k in self.hist.keys():
                self.hist[k].append(v)
            else:
                self.hist[k] = [v,]

    def _load_polar_data(self, polar_path:Path|str) -> Dict[str, np.ndarray]:
        """Function to read the polar data"""

        polar_txt = np.loadtxt(polar_path,skiprows=2)
        
        polar_data = {}
        polar_data["alpha"] = polar_txt[:,0]
        polar_data["Cl"]    = polar_txt[:,1]
        polar_data["Cd"]    = polar_txt[:,2]
        return polar_data
    
    def calculate_Cl(self, alpha:np.ndarray|float) -> np.ndarray|float:
        """Method to compute Cl at a given angle of attack"""
        if self.is_prop:
            return np.interp(alpha, 
                             self.polar_data["alpha"], 
                             self.polar_data["Cl"])
        else:
            return -np.interp(-alpha, 
                             self.polar_data["alpha"], 
                             self.polar_data["Cl"])
    
    def calculate_Cd(self, alpha:np.ndarray|float) -> np.ndarray|float:
        """Method to compute Cd at a given angle of attack"""
        if self.is_prop:
            return np.interp(alpha, 
                             self.polar_data["alpha"], 
                             self.polar_data["Cd"])
        else:
            return np.interp(-alpha, 
                             self.polar_data["alpha"], 
                             self.polar_data["Cd"])
        
    def calculate_induction(self, k:float, F:float) -> float:
        # applies Glauert correction for heavily loaded blades

        CT1 = -1.816
        a_cr = np.sqrt(np.abs(CT1))/2-1
        k_cr = a_cr/(1+a_cr)
        # CT_cr = CT1 - 2*np.sqrt(np.abs(CT1))

        if k < k_cr: # heavily loaded turbine
            a_ = 4*F*k
            b_ = -4*(np.sqrt(np.abs(CT1))-1)
            c_ = -CT1

            root = np.roots([a_, b_, c_])

            a = root[1]-1

            # a = (CT-CT1)/(4*(np.sqrt(np.abs(CT1))-1))-1
        else: # momentum theory result
            a = k/(1-k)
            # a = 1/2*(1-np.sqrt(CT+1))
        return a
    
    def calculate_residual(self, phi):
            
        # compute angles
        alpha_deg = self.beta - np.rad2deg(phi)

        # find forces
        Cl = self.calculate_Cl(alpha_deg)
        Cd = self.calculate_Cd(alpha_deg)

        # rotate forces
        Cx = Cl*np.cos(phi)-Cd*np.sin(phi)
        Cy = Cl*np.sin(phi)+Cd*np.cos(phi)
        
        # calculate the tip correction
        # F = tip_correction.ning_correction(self.r_R,self.r_R_H,self.B, phi)
        F = tip_correction.calculate_prandtl_correction3(
            B=self.B,
            phi=phi,
            r_R=self.r_R,
            r_R_H=self.r_R_H)
        
        # blade element momentum
        #   tangential
        kline = Cy*self.sig/(4*F*np.sin(phi)*np.cos(phi)) # Ning (Eq 42)
        aline = kline/(1+kline)
        
        #   axial
        k = Cx*self.sig/(4*F*(np.sin(phi))**2)
        a = self.calculate_induction(k, F) 

        # compute other coefficients
        Vy_Vx = self.TSR*self.r_R
        CT = Cx*self.sig*((1+a)/np.sin(phi))**2 # Ning (27)
        CQ = Cy*self.sig*((1-aline)/np.cos(phi))*((1+a)/np.sin(phi))*Vy_Vx # Ning (Eq 28) 
        CP = 2*np.pi*CQ
        Ca = Cy #is this azimuthal?

        residual = np.sin(phi)/(1+a) - 1/Vy_Vx*np.cos(phi)/(1-aline) # Ning (Eq 70)
            
        # save results
        res = dict(
            alpha = alpha_deg,
            Cl = Cl,
            Cd = Cd,
            Cx = Cx,
            Cy = Cy,
            F  = F,
            CT = CT,
            CP = CP,
            Ca = Ca,
            CQ = CQ,
            a = a,
            aline = aline,
            r = residual,
        )
        self._update_hist(res)

        return residual
    
    def solve(self):

        # Initial bound: 1st quadrant
        self.phi, res = optimize.brentq(self.calculate_residual, 
                                        self.phi_rng.min(), 
                                        self.phi_rng.max(), 
                                        full_output=True)

        # self.phi, res = optimize.newton(self.calculate_residual, 
        #                                 np.pi/10,
        #                                 maxiter=1000,
        #                                 # phi1, 
        #                                 full_output=True)


        print(res)
        # store results
        self.alpha = self.hist["alpha"][-1]
        self.Cl = self.hist["Cl"][-1]
        self.Cd = self.hist["Cd"][-1]
        self.Cx = self.hist["Cx"][-1]
        self.Cy = self.hist["Cy"][-1]
        self.F  = self.hist["F"][-1]
        self.CT = self.hist["CT"][-1]
        self.CP = self.hist["CP"][-1]
        self.Ca = self.hist["Ca"][-1]
        self.CQ = self.hist["CQ"][-1]
        self.a = self.hist["a"][-1]
        self.aline = self.hist["aline"][-1]


if __name__ == "__main__":
    
    N = 1000
    r_R_list = np.linspace(0.25,1,N)
    c_R_list = 0.18 - r_R_list * 0.06
    beta_list =  46+(35 - 50*r_R_list)
    B = 6
    J = 1.2
    idx = 500
    r_R = r_R_list[idx]
    c_R = c_R_list[idx]
    bet = beta_list[idx]

    ai = Annuli(polar_path=main_dir.joinpath("ARAD8pct_polar.txt"),
                r_R=r_R,
                c_R=c_R,
                beta=bet,
                B=B,
                J=J,
                r_R_H=0.25)

    # K = np.linspace(-1, 1000, 100000)
    # a = [ai.calculate_induction(ki, 1) for ki in K]
    print(r_R, ai.a, ai.aline, ai.alpha, np.rad2deg(ai.phi), ai.beta)

    fig, ax = plt.subplots()

    ax.plot(ai.hist["r"])
    ax.plot(ai.hist["Cl"])
    ax.plot(ai.hist["Cd"])
    ax.plot(ai.hist["CT"])

    # ax.plot(ai.hist["F"])
    # ax.plot(K, a)
    # ax.plot(ai.hist["a"])
    # ax.plot(ai.hist["aline"])

    plt.show()