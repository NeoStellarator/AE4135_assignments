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
                 Uinf: float,
                 Omega:float,
                 r1_R:float,
                 r2_R:float,
                 r_R_H:float,
                 R:float,
                 B:float,
                 c_R:float,
                 beta:float,
                 rho:float,
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
        self.rho=rho
        # iteration initialization
        self.a0     = a0
        self.aline0 = aline0

        self.hist:Dict[str,List[float]] = dict()
        
        # read & store polar data
        self.polar_data=self._load_polar_data(polar_path)
      
        # perform the iteration
        self.run_iteration()
        self.polar_data = self._load_polar_data(polar_path)

        # search only the 1st quadrant (Vx > 0, Vy > 0)
        self.phi_eps = 1e-6
        self.phi_rng = np.array([self.phi_eps, np.pi/2 - self.phi_eps])

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
        return np.interp(alpha,self.polar_data["alpha"],self.polar_data["Cd"])
    def ainduction(self, CT):

        CT1=1.816

        CT2=2*np.sqrt(CT1) -CT1
        # print(CT)
        if CT>=CT2:
            a = 1 + (CT-CT1 )/(4*(np.sqrt(CT1)-1))
        if CT<CT2:
            a = 0.5-0.5 *np.sqrt(1-CT)
        return a
    
    def run_iteration(self, tol=1e-5, iter_max=1e5):
        a = self.a0
        aline = self.aline0

        if k < k_cr: # heavily loaded turbine
            a_ = 4*F*k
            b_ = -4*(np.sqrt(np.abs(CT1))-1)
            c_ = -CT1

            root = np.roots([a_, b_, c_])

        while (max(np.abs(a-a_old)/(a_old if a_old !=0 else 1), 
                  np.abs(aline-aline_old)/(aline_old if aline_old !=0 else 1)) > tol
                and i<iter_max):
            Ux = self.Uinf*(1+a)
            Uy = (1-aline)*self.Omega*self.r_R*self.R
            # compute angles
            # phi = np.arctan((1/(self.TSR*self.r_R))*(1+a)/(1-aline))
            phi = np.arctan2(Ux,Uy)
            if self.isPropeller:
                alpha_deg =self.beta - np.rad2deg(phi)
            else:
                alpha_deg = (np.rad2deg(phi) - self.beta)

        # find forces
        Cl = self.calculate_Cl(alpha_deg)
        Cd = self.calculate_Cd(alpha_deg)

        # rotate forces
        Cx = Cl*np.cos(phi)-Cd*np.sin(phi)
        Cy = Cl*np.sin(phi)+Cd*np.cos(phi)
        
        # calculate the tip correction
        F = tip_correction.ning_correction(self.r_R,self.r_R_H,self.B, phi)
        # F = tip_correction.calculate_prandtl_correction3(
        #     B=self.B,
        #     phi=phi,
        #     r_R=self.r_R,
        #     r_R_H=self.r_R_H)
        
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
            
            # compute the new induction factors
            # RHS_1 = self.sig/(4*np.sin(phi)**2)*Cx
            # apply hub/tip loss correction
            f = tip_correction.calculate_prandtl_correction2(
                B=self.B,
                TSR=self.TSR,
                a=abs(a),
                a_line = abs(aline),
                r_R=self.r_R,
                r_R_H=self.r_R_H)
            # print(f)
            RHS_2 = self.sig/(4*f*np.sin(phi)*np.cos(phi))*Cy

            # a_new = RHS_1/(1+RHS_1)
            aline_new = RHS_2/(1+ RHS_2)
            CT = Cx*self.sig/(f*np.sin(phi)**2) #This definition was different

            a_new = self.ainduction(CT)

            
            
            # a_new /= f
            # aline_new /= f 

        self.phi, res = optimize.newton(self.calculate_residual, 
                                        np.mean(self.phi_rng),
                                        maxiter=1000,
                                        # phi1, 
                                        full_output=True)

            if not res.converged:
                print("not converged!!")

        # recompute and store the final converged state
        self.calculate_residual(self.phi)
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

        # print(i)
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
    def calculate_forces(self):
        Ux = self.Uinf*(1+self.a)
        Uy = (1-self.aline)*self.Omega*self.r_R*self.R
        w2 = Ux**2+Uy**2
        thrust = 0.5 * self.Cx * self.rho * w2 * self.chord * (self.r2_R-self.r1_R)*self.R * self.B
        torque = 0.5 * self.Cy * self.rho * w2 * self.chord * (self.r2_R-self.r1_R)*self.R * self.B * self.r_R*self.R
        return [thrust, torque]


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