import matplotlib.pyplot as plt
import numpy as np
from cache.AnnularIterator import AnnularIterator


ai = AnnularIterator("ARAD8pct_polar.txt")
def test_get_cl():
    assert ai.calculate_cl(0) == 0.5480
def test_get_cd():
    assert ai.calculate_cd(0) == 0.01039
def test_prandtl_correction():
    plt.figure()
    r_R_list = np.linspace(0,1,30)
    a = 1/3
    B = 3
    J = 1.2
    f_list = [ai.calculate_prandtl_correction(r_R,a,B,J) for r_R in r_R_list]
    print(f_list[-1])
    plt.plot(r_R_list,f_list)
    plt.show()
    assert True
def test_annular_iteration():
    N = 30
    r_R_list = np.linspace(0.25,1,30)
    c_R_list = 0.18 - r_R_list * 0.06
    beta_list =  35 - 50*r_R_list
    B = 6
    J=1.2
    r_R = r_R_list[0]
    solidity_r_list = B/2*np.pi*c_R_list/r_R_list
    a_0 = 1/3
    a_line_0 = 1/3
    beta=beta_list[0]
    sigma_r = solidity_r_list[0]
    a, a_line = ai.run_iteration(J,B,r_R, a_0,a_line_0, beta,sigma_r,tolerance = 1e-4)
    
