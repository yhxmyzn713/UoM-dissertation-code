import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2

def marcumq(a, b):
    return ncx2.sf(b**2, 2, a**2)

def calculate_theoretical_pd(PT, gamma_C, rate, gamma_th, gamma_T_dB):
    Pr = PT - (2 ** rate - 1) / gamma_C
    if Pr < 0:
        return 0
    gamma_T = 10 ** (gamma_T_dB / 10)
    return marcumq(np.sqrt(2 * Pr * gamma_T), np.sqrt(2 * gamma_th))

def main():
    # System parameters
    Pt = 20
    gamma_c_dB = 0
    gamma_C = 10 ** (gamma_c_dB / 10)
    PFA_target = 0.01
    gamma_th = -np.log(PFA_target)
    gamma_T_dBs = [-10, -5, 0, 5]
    sigma = 1
    L = 10
    gamma_R_dB = 10
    gamma_R = 10 ** (gamma_R_dB / 10)
    rate_range = np.linspace(0, 4.2, 24)
    #num_iterations = 10000

    # Simulation and theoretical calculation
    pd_Pr_values = {}
    for gamma_T in gamma_T_dBs:
        pd_Pr_values[gamma_T] = [calculate_theoretical_pd(Pt, gamma_C, rate, gamma_th, gamma_T) for rate in rate_range]

    # Plotting results
    plt.figure(figsize=(8, 6))
    for gamma_T in gamma_T_dBs:
        plt.plot(rate_range, pd_Pr_values[gamma_T], 'o-', label=f'γ_T = {gamma_T} dB')

    plt.xlabel('Rate (bpcu)', fontsize=15)
    plt.ylabel('PD',fontsize=15)
    #plt.title('PD vs Rate for Different γ_T Values')
    plt.legend()
    plt.xlim(0, 4.5)
    plt.tick_params(labelsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('PD_vs_Rate(gamma_T).png')
    plt.show()

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script finished")
