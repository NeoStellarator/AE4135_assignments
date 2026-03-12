import numpy as np
from scipy.optimize import fsolve
class AnnularIterator3:
    def __init__(self,Uinf,J,B,R,C_r,Beta,r_R,polar_path):
        self.Uinf = Uinf
        self.J = J
        self.B = B
        self.R = R
        self.C_r = C_r
        self.Beta = Beta
        self.r_R = r_R

        self.rho = 1.006
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
    def calculate_csolidity(self,B,chord,r_R):
        return B*chord/(2*np.pi*r_R*self.R)
    def calculate_a(self,CT):
        CT1 = 1.816
        CT2 = 2*np.sqrt(CT1)-CT1
        if CT<CT2:
            a = 1/2 - np.sqrt(1-CT)/2
        else:
            a= 1+(CT-CT1)/(4*np.sqrt(CT1)-4)
        return a
    # def calculate_a_line(self,Phi,Ftan,sigma_r):
    #     def func_a_line(a_line):
    #         return a_line/(1+a_line)-sigma_r/(4*np.cos(Phi)*np.sin(Phi))*Ftan
    #     return fsolve(func_a_line,0)[0]
    
    def calculate_a_line(self,Phi,Cy,sigma_r):
        A = sigma_r/(4*np.cos(Phi)*np.sin(Phi))*Cy
        return A/(1-A)

    
    def calculate_prandtl_correction(self,r_R,a,B,J):
        gamma = np.pi/J
        d = 2*np.pi/B*(1-a)/np.sqrt(gamma**2+(1-a)**2)
        f_tip = 2/np.pi*np.arccos(np.exp((-np.pi*(1-r_R)/d)))

        f_root = 2/np.pi*np.arccos(np.exp((-np.pi*(r_R)/d))) #reversed
        return f_tip*f_root
        
    def run_iteration(self,tolerance = 1e-4):
        a_0 = 0.3
        a_line_0 = 0
        a_diff = 1e6
        a_line_diff = 1e6
        chord = self.C_r*self.R

        a = a_0
        a_line = a_line_0
        
        while a_diff>tolerance and a_line_diff>tolerance:
            Urot = self.Uinf * (1-a)
            Omega = 2*np.pi*self.Uinf/(2*self.R*self.J)
            Utan = (1+a_line)*Omega*r_R*self.R
            Umag2 = np.sqrt(Urot**2+Utan**2)
            Phi2 = np.arctan2(Urot,Utan)
            Phi = np.arctan(self.J/np.pi*(1-a)/(1+a_line))
            print(Phi*180/np.pi, Phi2*180/np.pi)
            alpha = self.Beta+Phi*180/np.pi
            Cl = self.calculate_cl(alpha)
            Cd = self.calculate_cd(alpha)
            sigma_r = self.calculate_csolidity(self.B,chord,self.r_R)
            Cx = Cl*np.cos(Phi)+Cd*np.sin(Phi)
            Cy = Cl*np.sin(Phi)-Cd*np.cos(Phi)
            Faxial = 0.5*self.rho*chord*Umag2*chord*Cy
            Fazim = 0.5*self.rho*chord*Umag2*chord*Cx

            
            area = np.pi*self.R**2
            CT = Faxial*B/(0.5*area*self.Uinf**2)
            print("CT:", CT)
            prandtl = self.calculate_prandtl_correction(self.r_R,a,self.B,self.J)
            a_new = self.calculate_a(CT)/prandtl
            a_line_new = self.calculate_a_line(Phi,Cy,sigma_r)/prandtl
            a_diff = abs(a_new-a)
            a_line_diff = abs(a_line_new-a_line)
            a = a*0.75+a_new*0.25
            a_line = a_line*0.75+a_line_new*0.25
            CT2 = 4*a*(1-a)
            print("CT2:", CT2)
        return a, a_line


            

if __name__ == "__main__":
    
    N = 30
    r_R_list = np.linspace(0.25,1,30)
    dr_R = r_R_list[2]-r_R_list[1]
    c_R_list = 0.18 - r_R_list * 0.06
    beta_list =  35 - 50*r_R_list
    B = 6
    #Use J in code
    J=1.2
    idx = 15
    r_R = r_R_list[idx]
    print("r/R:", r_R)
    solidity_r_list = B/(2*np.pi)*c_R_list/r_R_list
    a_0 = -0.3
    a_line_0 = 0
    beta=beta_list[idx]
    c_R = c_R_list[idx]
    sigma_r = solidity_r_list[idx]
    ai = AnnularIterator3(Uinf=60, J = J, R=0.7,B=B, C_r=c_R, Beta=beta, r_R=r_R,polar_path="1_BEM/ARAD8pct_polar.txt")
    a, a_line = ai.run_iteration()
    print(a, a_line)
    # print(a, a_line)
    