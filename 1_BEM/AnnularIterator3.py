import numpy as np
class AnnularIterator3:
    def __init__(self,Uinf,Utan,Omega,R,polar_path):
        self.Uinf = Uinf
        self.Utan = Utan
        self.Omega = Omega
        self.R = R
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
    def calculate_a(self,CT):
        CT1 = 1.816
        CT2 = 2*np.sqrt(CT1)-CT1
        if CT<CT2:
            a = 1/2 - np.sqrt(1-CT)/2
        else:
            a= 1+(CT-CT1)/(4*np.sqrt(CT1)-4)
        return a
    def calculate_prandtl_correction(self,r_R,a,B,gamma):
        d = 2*np.pi/B*(1-a)/np.sqrt(gamma**2+(1-a)**2)
        f_tip = 2/np.pi*np.arccos(np.exp((-np.pi*(1-r_R)/d)))

        f_root = 2/np.pi*np.arccos(np.exp((-np.pi*(r_R)/d))) #reversed
        return f_tip*f_root
    def run_iteration(self,r_R,chord,dr_R,B, beta,tolerance = 1e-4):
        a_0 = 0.3
        a_line_0 = 0
        a_diff = 1e6
        a_line_diff = 1e6


        a = a_0
        a_line = a_line_0
        
        while a_diff>tolerance or a_line_diff>tolerance:
            Urot = self.Uinf * (1-a)
            Utan = (1+a_line)*self.Omega*r_R*self.R
            Umag2 = Urot**2+Utan**2
            Phi = np.arctan2(Urot,Utan)
            alpha = beta+Phi*180/np.pi
            Cl = self.calculate_cl(alpha)
            Cd = self.calculate_cd(alpha)
            L = 0.5*Umag2*Cl*chord
            D = 0.5*Umag2*Cd*chord
            Fnorm = L*np.cos(Phi)+D*np.sin(Phi)
            Ftan = L*np.sin(Phi)-D*np.cos(Phi)
            gamma = 0.5*np.sqrt(Umag2)*Cl*chord

            Faxial = Fnorm*self.R*(dr_R)*B
            area = np.pi*self.R**2
            CT = Faxial/(0.5*area*self.Uinf**2)
            prandtl = self.calculate_prandtl_correction(r_R,a,B,self.Omega*self.R/self.Uinf)
            a_new = self.calculate_a(CT)/prandtl


            

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
    ai = AnnularIterator3("ARAD8pct_polar.txt")
    a, a_line = ai.run_iteration(J,B,r_R, a_0,a_line_0, beta,sigma_r,tolerance = 1e-4)
    print(a, a_line)