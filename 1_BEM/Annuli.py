from typing import Dict, List
import numpy as np
from pathlib import Path

from globals import main_dir
import tip_correction
from scipy import optimize

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
        self.polar_data=self._load_polar_data(polar_path)

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

    def calculate_induction(self, CT):
        # applies Glauert correction for heavily loaded blades

        CT1 = -1.816
        CT_cr = CT1 - 2*np.sqrt(np.abs(CT1))
        if CT < CT_cr: # heavily loaded turbine
            a = (CT-CT1)/(4*(np.sqrt(np.abs(CT1))-1))-1
        else: # momentum theory result
            a = 1/2*(1-np.sqrt(CT+1))
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
        k = Cy*self.sig/(4*F*np.sin(phi)*np.cos(phi)) # Ning (Eq 42)
        aline = k/(1+k)
        
        #   axial
        CT = Cx*self.sig*((1+a)/np.sin(phi))**2 # Ning (27)
        a = self.calculate_induction(CT) 

        # compute other coefficients
        Vy_Vx = self.TSR*self.r_R
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
        )
        self._update_hist(res)
        return residual
    
    def solve(self):

        # Initial bound: 1st quadrant
        phi0 = 1E-6
        phi1 = np.pi/2
        self.phi = optimize.brentq(self.calculate_residual, phi0, phi1)

        # store results
        self.alpha = self.temp_res["alpha"]
        self.Cl = self.temp_res["Cl"]
        self.Cd = self.temp_res["Cd"]
        self.Cx = self.temp_res["Cx"]
        self.Cy = self.temp_res["Cy"]
        self.F  = self.temp_res["F"]
        self.CT = self.temp_res["CT"]
        self.CP = self.temp_res["CP"]
        self.Ca = self.temp_res["Ca"]
        self.CQ = self.temp_res["CQ"]
        self.a = self.temp_res["a"]
        self.aline = self.temp_res["aline"]


if __name__ == "__main__":
    
    N = 30
    r_R_list = np.linspace(0.25,1,30)
    c_R_list = 0.18 - r_R_list * 0.06
    beta_list =  35 - 50*r_R_list
    B = 6
    J = 1.2
    idx = 15
    r_R = r_R_list[idx]
    c_R = c_R_list[idx]
    bet = beta_list[idx]

    ai = Annuli(polar_path="ARAD8pct_polar.txt",
                r_R=r_R,
                c_R=c_R,
                beta=bet,
                B=B,
                J=J,
                r_R_H=0.25)

    print(ai.a, ai.aline)
