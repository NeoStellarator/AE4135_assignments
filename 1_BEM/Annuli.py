from typing import Dict
import numpy as np
from pathlib import Path

from globals import main_dir
import tip_correction

class Annuli:
    def __init__(self, 
                 polar_path:Path|str, 
                 Uinf: float,
                 Omega:float,
                 r1_R:float,
                 r2_R:float,
                 r_R_H:float,
                 R:float,
                 B:float,
                 c_R:float,
                 beta:float,
                 a0:float=0,
                 aline0:float=0,
                 isPropeller:bool = False):

        self.Uinf = Uinf
        self.r1_R = r1_R
        self.r2_R = r2_R
        self.r_R_H = r_R_H
        self.Omega = Omega
        self.R=R
        self.B = B
        self.c_R = c_R
        self.beta = beta
        self.TSR = (Omega * R) / Uinf
        self.r_R = (r1_R + r2_R) / 2
        self.sig = (B * c_R) / (2* np.pi * self.r_R)
        self.chord = c_R*R
        self.A = np.pi*((r2_R *R )**2- (r1_R*R)**2)
        self.isPropeller = isPropeller

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
    def ainduction(self, CT):
        """
        This function calculates the induction factor 'a' as a function of thrust coefficient CT 
        including Glauert's correction
        """
        CT1=1.816
        CT2=2*np.sqrt(CT1)-CT1
        # print(CT)
        if CT>=CT2:
            a = 1 + (CT-CT1)/(4*(np.sqrt(CT1)-1))
        if CT<CT2:
            a = 0.5-0.5*np.sqrt(1-CT)
        return a
    
    def run_iteration(self, tol=1e-5, iter_max=1e5):
        a = self.a0
        aline = self.aline0

        a_old = 1.1*(a+1)
        aline_old = 1.1*(aline+1)

        i=1

        while (max(np.abs(a-a_old)/(a_old if a_old !=0 else 1), 
                  np.abs(aline-aline_old)/(aline_old if aline_old !=0 else 1)) > tol
                and i<iter_max):
            Ux = self.Uinf*(1+a)
            Uy = (1-aline)*self.Omega*self.r_R*self.R
            # compute angles
            phi = np.arctan((1/(self.TSR*self.r_R))*(1+a)/(1-aline))
            if self.isPropeller:
                alpha_deg =self.beta - np.rad2deg(phi)
            else:
                alpha_deg = (np.rad2deg(phi) - self.beta)

            # find forces
            Cl = self.calculate_Cl(alpha_deg)
            Cd = self.calculate_Cd(alpha_deg)

            # rotate forces
            Cx = Cl*np.cos(phi)+Cd*np.sin(phi)
            Cy = Cl*np.sin(phi)-Cd*np.cos(phi)
            
            # compute the new induction factors
            # RHS_1 = self.sig/(4*np.sin(phi)**2)*Cx
            # apply hub/tip loss correction
            f = tip_correction.calculate_prandtl_correction_3(
                B=self.B,
                TSR=self.TSR,
                a=abs(a),
                a_line = abs(aline),
                r_R=self.r_R,
                r_R_H=self.r_R_H)
            # print(f)
            RHS_2 = self.sig/(4*f*np.sin(phi)*np.cos(phi))*Cy

            # a_new = RHS_1/(1+RHS_1)
            aline_new = RHS_2/(1+   RHS_2)
            CT = Cx*self.sig/(f*np.sin(phi)**2) #This definition was different

            a_new = self.ainduction(CT)

            
            
            # a_new /= f
            # aline_new /= f 

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
        self.Ct = CT
        self.a = a
        self.aline = aline
        self.Ux = Ux
        self.Uy= Uy


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
