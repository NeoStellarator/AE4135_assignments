from typing import Dict
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
                 r_R_H:float=0):

        # save blade 
        self.r_R   = r_R   # blade radial coordinate
        self.r_R_H = r_R_H # hub radial coordinate
        self.c_R   = c_R   # blade chord 
        self.beta  = beta  # blade twist
        self.B     = B     # blade number
        self.J     = J     # advance ratio
        self.TSR   = np.pi/J # tip speed ratio
        self.sig   = self.B/(2*np.pi)*self.c_R/self.r_R # blade solidity

        # read & store polar data
        self.polar_data=self._load_polar_data(polar_path)

        self.phi = self.solve()


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
        return np.interp(alpha,self.polar_data["alpha"],self.polar_data["Cl"])
    
    def calculate_Cd(self, alpha:np.ndarray|float) -> np.ndarray|float:
        """Method to compute Cd at a given angle of attack"""
        return np.interp(alpha,self.polar_data["alpha"],self.polar_data["Cd"])

    def calculate_residual(self, phi):
            
        # compute angles
        alpha_deg = self.beta - np.rad2deg(phi)

        # find forces
        Cl = self.calculate_Cl(alpha_deg)
        Cd = self.calculate_Cd(alpha_deg)

        # rotate forces Ning 19
        Cx = Cl*np.cos(phi)-Cd*np.sin(phi)
        Cy = Cl*np.sin(phi)+Cd*np.cos(phi)
        
        F = tip_correction.ning_correction(self.r_R,self.r_R_H,self.B, phi)
        RHS_1 = self.sig*Cx/(4*F*np.sin(phi)**2) #Ning (33)
        RHS_2 = Cy*self.sig/(4*F*np.sin(phi)*np.cos(phi)) #Ning (42)
        
        a = RHS_1/(1-RHS_1) # Ning (34) must be replaced by the correction
        aline = RHS_2/(1+RHS_2) # Ning (43)

        # compute other coefficients
        velocity_ratio = (1+a)/((1-aline)*np.tan(phi)) #Vy/Vx derived from the expression W = Vx(1+a)/sin(phi) = Vy(1+a')/cos(phi) Ning 25 and 26
        CT = Cx*self.sig*((1+a)/np.sin(phi))**2 # Ning (27)
        CQ = Cy*self.sig*((1-aline)/np.cos(phi))*((1+a)/np.sin(phi))*velocity_ratio # Ning (28) 
        CP = 2*np.pi*CQ
        Ca = Cy #is this azimuthal?

        residual = np.sin(phi)/(1+a) - 1/velocity_ratio*np.cos(phi)/(1-aline) # Ning (70)
            
        # save converged results
        self.alpha = alpha_deg
        self.Cl = Cl
        self.Cd = Cd
        self.Cx = Cx
        self.Cy = Cy
        self.F  = F
        self.CT = CT
        self.CP = CP
        self.Ca = Ca
        self.CQ = CQ
        self.a = a
        self.aline = aline
        return residual
    def solve(self):
        phi = optimize.brentq(self.calculate_residual,1E-6,np.pi/2) #First quadrant 
        return phi

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
