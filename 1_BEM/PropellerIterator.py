import numpy as np
# from AnnularIterator import AnnularIterator
import pandas as pd
from AnnularIterator2 import AnnularIterator2
import matplotlib.pyplot as plt
class PropellerIterator:
    def __init__(self,B,start_r_R,N):
        self.B = B
        self.start_r_R = start_r_R
        self.N = N
        self.r_R_list = np.linspace(0.25,1,30)
        self.c_R_list = 0.18 - self.r_R_list * 0.06
        self.beta_list =  35 - 50*self.r_R_list
        self.solidity_r_list = B/(2*np.pi)*self.c_R_list/self.r_R_list
        self.dr_R = (1-start_r_R)/(self.N-1)
        self.annularIterator = AnnularIterator2("ARAD8pct_polar.txt")
    def spanwise_induced(self,J):
        a_list = []
        a_line_list = []
        Cl_list = []
        Cd_list = []
        phi_deg_list = []
        alpha_list = []
        Cx_list = []
        Cy_list = []
        beta_list = []
        f_list = []
        CT_list = []
        for idx in range(len(self.r_R_list)):
            r_R = self.r_R_list[idx]
            c_R = self.c_R_list[idx]
            beta = self.beta_list[idx]
            sigma_r = self.solidity_r_list[idx] 
            a,a_line,phi_deg,alpha, Cl, Cd, Cx, Cy, beta,f,CT = self.annularIterator.run_iteration(J=J,B=self.B,r_R=r_R,a_0=0.3, a_line_0=0,beta=beta,sigma_r=sigma_r)
            a_list.append(a)
            a_line_list.append(a_line)
            Cl_list.append(Cl)
            Cd_list.append(Cd)
            Cx_list.append(Cx)
            Cy_list.append(Cy)
            beta_list.append(beta)
            phi_deg_list.append(phi_deg)
            alpha_list.append(alpha)
            f_list.append(f)
            CT_list.append(CT)
            
        debug_df = pd.DataFrame({
            'phi_deg': phi_deg_list,
            'alpha': alpha_list,
            'Cl': Cl_list,
            'Cd': Cd_list,
            'Cx': Cx_list,
            'Cy': Cy_list,
            'beta':beta_list,
            'a':a_list,
            'a_line':a_line_list,
            'f':f_list,
            'CT':CT_list
            
        })    
        return a_list, a_line_list,debug_df
    
if __name__ == "__main__":
    pi = PropellerIterator(6,0.25,100)
    a_list, a_line_list,debug_df = pi.spanwise_induced(1.2)
    # plt.figure()
    # plt.plot(pi.r_R_list,a_list)
    # plt.plot(pi.r_R_list,a_line_list)
    # plt.show()
    fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    axes = axes.flatten()

    columns = ['phi_deg', 'alpha', 'Cl', 'Cd', 'Cx', 'Cy','beta','a','a_line','f','CT']
    for idx, col in enumerate(columns):
        axes[idx].plot(pi.r_R_list, debug_df[col])
        axes[idx].set_xlabel('r_R')
        axes[idx].set_ylabel(col)
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()