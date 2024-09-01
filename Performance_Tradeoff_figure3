import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from scipy.stats import ncx2

def marcumq(a, b):
    return ncx2.sf(b**2, 2, a**2)

def calculate_theoretical_pd(PT, gamma_C, rate, gamma_th, gamma_T_dB):
    Pr = PT - (2 ** rate - 1) / gamma_C
    if Pr < 0:
        return 0
    gamma_T = 10 ** (gamma_T_dB / 10)
    return marcumq(np.sqrt(2 * Pr * gamma_T), np.sqrt(2 * gamma_th))

def simulate_PD(L, num_iterations, sigma, PT, gamma_C, rate, gamma_th, gamma_R, gamma_T_dB):
    PD_count = 0
    pr = PT - (2 ** rate - 1) / gamma_C
    if pr < 0:
        return np.nan

    gamma_r_linear = 10 ** (gamma_R / 10)
    gamma_t_linear = 10 ** (gamma_T_dB / 10)
    gamma_d = np.sqrt(gamma_r_linear * sigma**2)
    for _ in range(num_iterations):
        n_d, n_s = generate_noise(L, sigma)
        sr = np.sqrt(pr) * np.ones(L) / np.sqrt(L)
        x_d = gamma_d * sr + n_d
        x_s = gamma_t_linear * sr + n_s
        A = np.outer(x_d, x_d.conj()) + np.outer(x_s, x_s.conj())
        lamda_max_A = eigvalsh(A).max()
        glrt = (lamda_max_A - np.linalg.norm(x_d) ** 2) / (sigma ** 2)
        if glrt >= gamma_th:
           PD_count += 1
    return PD_count / num_iterations

def generate_noise(L, sigma):
    return (np.sqrt(sigma / 2) * (np.random.randn(L) + 1j * np.random.randn(L)),
            np.sqrt(sigma / 2) * (np.random.randn(L) + 1j * np.random.randn(L)))

def main():
    PFA_target = 0.01
    gamma_th = -np.log(PFA_target)
    L = 10
    num_iterations = 10000
    sigma = 1
    PT = 10
    gamma_C_dB = 0
    gamma_C = 10 ** (gamma_C_dB / 10)
    gamma_T_dB = 0

    gamma_Rs = [-10, 10, 20]
    rates = np.linspace(0, 4.2, 24)
    plt.figure(figsize=(8, 6))

    theoretical_pds = [calculate_theoretical_pd(PT, gamma_C, rate, gamma_th, gamma_T_dB) for rate in rates]
    plt.plot(rates, theoretical_pds, 'k-', label='Theoretical.Approx PD')

    for gamma_R in gamma_Rs:
        results = []
        for rate in rates:
            PD = simulate_PD(L, num_iterations, sigma, PT, gamma_C, rate, gamma_th, gamma_R, gamma_T_dB)
            results.append(PD)
        plt.plot(rates, results, 'o-', label=f'Sim.γ_R = {gamma_R} dB')

    plt.xlabel('Rate (bpcu)', fontsize=15)
    plt.ylabel('PD', fontsize=15)
    # plt.title('PD vs. Rate for Different γ_R Values')
    plt.ylim(0, 1)
    plt.xlim(0, 3.5)
    plt.tick_params(labelsize=12)
    plt.legend( )
    plt.grid(True, which="both", ls="-")
    plt.show()

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script finished")
