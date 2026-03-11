import numpy as np
import pandas as pd
class AnnularIterator2:
    def __init__(self,polar_path):
        self.polar_data=self.load_polar_data(polar_path)
    def load_polar_data(self, polar_path):
        polar_txt=np.loadtxt(polar_path,skiprows=2)
        polar_data = {}
        polar_data["alpha"]= polar_txt[:,0]
        polar_data["cl"]= polar_txt[:,1]
        polar_data["cd"]= polar_txt[:,2]
        return polar_data
    def calculate_cl(self,alpha):
        return np.interp(alpha,self.polar_data["alpha"],self.polar_data["cl"])
    def calculate_cd(self,alpha):
        return np.interp(alpha,self.polar_data["alpha"],self.polar_data["cd"])
    def calculate_prandtl_correction(self,r_R,a,B,J):
        gamma = np.pi/J
        d = 2*np.pi/B*(1-a)/np.sqrt(gamma**2+(1-a)**2)
        f_tip = 2/np.pi*np.arccos(np.exp((-np.pi*(1-r_R)/d)))

        # f_root = 2/np.pi*np.arccos(np.exp((-np.pi*(r_R)/d))) #reversed
        return f_tip

    # def run_iteration(self,J,B,r_R, a_0,a_line_0, beta,sigma_r,tolerance = 1e-4):
    #     a = a_0
    #     a_line = a_line_0
    #     a_diff = 1e6
    #     a_line_diff = 1e6
    #     while a_diff>tolerance or a_line_diff>tolerance:
            

    #         print("what")


    #         # f=self.calculate_prandtl_correction(r_R,a,B,J)
    #         phi = np.arctan2(J/(np.pi*r_R)*(1-a)/(1+a_line))
    #         phi_deg = phi*180/np.pi
    #         alpha = phi_deg-beta
            
    #         Cl = self.calculate_cl(alpha)
    #         Cd = self.calculate_cd(alpha)

    #         Cx = Cl*np.cos(phi)+Cd*np.sin(phi)
    #         Cy = Cl*np.sin(phi)-Cd*np.cos(phi)


    #         RHS_1 = sigma_r/(4*np.sin(phi)**2)*Cx
    #         RHS_2 = sigma_r/(4*np.sin(phi)*np.cos(phi))*Cy

    #         a_new = RHS_1/(1+RHS_1)
    #         a_line_new = RHS_2/(1-RHS_2)

    #         a_diff = abs(a-a_new)
    #         a_line_diff = abs(a_line-a_line_new)
    #         a = a_new*0.25 + a*0.75
    #         a_line = a_line_new*0.25 + a_line*0.75
    #     return a, a_line
    def calculate_CT(self,a):
        CT1 = 1.816
        boundary = 1-np.sqrt(CT1)/2
        if a<boundary:
            return 4*a*(1-a)
        else:
            return CT1 - 4*(np.sqrt(CT1)-1)*(1-a)
    def run_iteration(self,J,B,r_R, a_0,a_line_0, beta,sigma_r,tolerance = 1e-6):
        a = a_0
        a_line = a_line_0
        a_diff = 1e6
        a_line_diff = 1e6
        while a_diff>tolerance or a_line_diff>tolerance:
            
            f=self.calculate_prandtl_correction(r_R,a,B,J)
            
            phi = np.arctan(J/(np.pi*r_R)*(1-a/f)/(1+a_line/f))
            phi_deg = phi*180/np.pi
            alpha = phi-beta
            Cl = self.calculate_cl(alpha)
            Cd = self.calculate_cd(alpha)

            Cx = Cl*np.cos(phi)+Cd*np.sin(phi)
            Cy = Cl*np.sin(phi)-Cd*np.cos(phi)
            RHS_1 = sigma_r/(4*np.sin(phi)**2)*Cx
            RHS_2 = sigma_r/(4*np.sin(phi)*np.cos(phi))*Cy

            a_new = RHS_1/(1+RHS_1)
            a_line_new = RHS_2/(1-RHS_2)
            
            a_diff = abs(a-a_new)
            a_line_diff = abs(a_line-a_line_new)
            a = a_new*0.25 + a*0.75
            a_line = a_line_new*0.25 + a_line*0.75
        CT = self.calculate_CT(a)
        return a, a_line,phi_deg,alpha, Cl, Cd, Cx, Cy,beta,f,CT
            
    # def run_iteration_propeller(self,J,B,r_R, a_0,a_line_0, beta,sigma_r,tolerance = 1e-4):
    #     a = a_0
    #     a_line = a_line_0
    #     a_diff = 1e6
    #     a_line_diff = 1e6
    #     while a_diff>tolerance or a_line_diff>tolerance:
            
    #         f=self.calculate_prandtl_correction(r_R,a,B,J)
    #         phi = np.arctan(J/np.pi*(1+a/f)/(1+a_line/f))
    #         phi_deg = phi*180/np.pi
    #         alpha = phi_deg-beta
            
    #         Cl = self.calculate_cl(alpha)
    #         Cd = self.calculate_cd(alpha)

    #         Cx = Cl*np.cos(phi)+Cd*np.sin(phi)
    #         Cy = Cl*np.sin(phi)-Cd*np.cos(phi)
    #         RHS_1 = sigma_r/(4*np.sin(phi)**2)*Cx
    #         RHS_2 = sigma_r/(4*np.sin(phi)*np.cos(phi))*Cy

    #         a_new = RHS_1/(1+RHS_1)
    #         a_line_new = RHS_2/(1-RHS_2)

    #         a_diff = abs(a-a_new)
    #         a_line_diff = abs(a_line-a_line_new)
    #         a = a_new*0.25 + a*0.75
    #         a_line = a_line_new*0.25 + a_line*0.75
    #     return a, a_line    

if __name__ == "__main__":
    
    N = 30
    r_R_list = np.linspace(0.25,1,30)
    c_R_list = 0.18 - r_R_list * 0.06
    beta_list =  35 - 50*r_R_list
    B = 6
    J=1.2
    idx = 15
    r_R = r_R_list[idx]
    solidity_r_list = B/(2*np.pi)*c_R_list/r_R_list
    a_0 = -0.3
    a_line_0 = 0
    beta=beta_list[idx]
    sigma_r = solidity_r_list[idx]
    ai = AnnularIterator("ARAD8pct_polar.txt")
    a, a_line = ai.run_iteration(J,B,r_R, a_0,a_line_0, beta,sigma_r,tolerance = 1e-4)
    print(a, a_line)
