from typing import Dict
import numpy as np
from pathlib import Path

from globals import main_dir
import tip_correction

class Annuli:

    def __init__(self, 
                 polar_path:Path|str, 
                 r_R:float,
                 c_R:float,
                 dr_R:float,
                 beta:float,
                 B:int, 
                 J:float,
                 R:float,
                 r_R_H:float=0,
                 a0:float=0,
                 aline0:float=0):

        # save blade 
        self.R     = R     # blade radius
        self.r_R   = r_R   # blade radial coordinate
        self.r_R_H = r_R_H # hub radial coordinate
        self.c_R   = c_R   # blade chord 
        self.beta  = beta  # blade twist
        self.dr_R  = dr_R  # blade element thickness 
        self.B     = B     # blade number
        self.J     = J     # advance ratio
        self.TSR   = np.pi/J # tip speed ratio
        self.sig   = self.B/(2*np.pi)*self.c_R/self.r_R # blade solidity

        # iteration initialization
        self.a0     = a0
        self.aline0 = aline0

        # read & store polar data
        self.polar_data=self._load_polar_data(polar_path)

        # perform the iteration
        self.run_iteration()

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
    def CTfunction(self,a, glauert = False):
        """
        This function calculates the thrust coefficient as a function of induction factor 'a'
        'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
        """
        CT = 4*a*(1-a)  
        if glauert:
            CT1=1.816
            a1=1-np.sqrt(CT1)/2
            if a>a1:
                CT = CT1-4*(np.sqrt(CT1)-1)*(1-a)
        
        return CT
    def ainduction(self, CT):
        """
        This function calculates the induction factor 'a' as a function of thrust coefficient CT 
        including Glauert's correction
        """
        a = np.zeros(np.shape(CT))
        CT1=1.816
        CT2=2*np.sqrt(CT1)-CT1
        if CT>=CT2:
            a = 1 + (CT-CT1)/(4*(np.sqrt(CT1)-1))
        if CT<CT2:
            a = 0.5-0.5*np.sqrt(1-CT)
        return a

    def run_iteration(self, tol=1e-6, iter_max=1e5):
        a = self.a0
        aline = self.aline0

        a_old = 1.1*(a+1)
        aline_old = 1.1*(aline+1)

        i=1

        while (max(np.abs(a - a_old), np.abs(aline - aline_old)) > tol
       and i < iter_max):
            
            # compute angles
            phi = np.arctan((1/(self.TSR*self.r_R))*(1-a)/(1+aline))
            alpha_deg =-(self.beta - np.rad2deg(phi))

            # find forces
            Cl = self.calculate_Cl(alpha_deg)
            Cd = self.calculate_Cd(alpha_deg)

            lift = 0.5*vmag2*cl*chord
            drag = 0.5*vmag2*cd*chord

            # rotate forces
            Cx = (Cl*np.cos(phi)+Cd*np.sin(phi))
            Cy = (Cl*np.sin(phi)-Cd*np.cos(phi))
            
            # compute other coefficients
            Ct = Cx*self.sig*((1-a)/np.sin(phi))**2
            Ca = Cy*self.sig*((1-a)/np.sin(phi))**2
            Cq = Ca*self.r_R
            Cp = Cx*self.sig*((1-a)/np.sin(phi))**3

            CT = Cx

            a= self.ainduction(CT)
            

            # compute the new induction factors
            # RHS_1 = self.sig/(4*np.sin(phi)**2)*Cx
            # RHS_2 = self.sig/(4*np.sin(phi)*np.cos(phi))*Cy

            
            # a_new = RHS_1/(1+RHS_1)
            # # Ct = self.CTfunction(a_new,glauert=True)
            # a_new = self.ainduction(Ct)
            # aline_new = RHS_2/(1-RHS_2)
            
            # apply hub/tip loss correction
            f = tip_correction.calculate_prandtl_correction(
                B=self.B,
                TSR=self.TSR,
                a=a,
                a_line = aline,
                r_R=self.r_R,
                r_R_H=self.r_R_H)
            
            
            a_new /= f
            aline_new /= f 

            # update the values of a
            a_old = a
            aline_old = aline
                        
            a = min(a_new*0.1 + a*0.9, 0.95)
            aline = aline_new*0.1 + aline*0.9

            # update iteration count
            i += 1

        print(i)
        # save converged results
        self.phi = np.rad2deg(phi)
        self.alpha = alpha_deg
        self.Cl = Cl
        self.Cd = Cd
        self.Cx = Cx
        self.Cy = Cy
        self.f  = f
        self.Ct = Ct
        self.Cp = Cp
        self.Ca = Ca
        self.Cq = Cq
        self.a = a
        self.aline = aline

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

    ai = Annuli(polar_path=main_dir.joinpath("ARAD8pct_polar.txt"),
                r_R=r_R,
                c_R=c_R,
                dr_R=0.01,
                beta=bet,
                B=B,
                J=J,
                r_R_H=0.25)

    print(ai.a, ai.aline)
