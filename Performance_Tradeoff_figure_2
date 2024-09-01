import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

def simulate_PFA(L, num_iterations, sigma, PT, gamma_C, rate, gamma_th, gamma_R):
    PFA_count = 0
    gamma_c_linear = 10 ** (gamma_C / 10)  # Convert gamma_C from dB to linear scale
    gamma_r_linear = 10 ** (gamma_R / 10)  # Convert gamma_R from dB to linear scale
    pr = PT - (2 ** rate - 1) / gamma_c_linear  # Calculate received power using Shannon's formula

    if pr < 0:  # Handle negative power scenario gracefully
        return np.nan  # Return NaN or handle appropriately

    gamma_d = np.sqrt(gamma_r_linear)  # Calculate gamma_d from gamma_R
    for _ in range(num_iterations):
        n_d = np.sqrt(sigma) * (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)
        n_s = np.sqrt(sigma) * (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)
        sr = np.sqrt(pr) * np.ones(L) / np.sqrt(L)
        x_d = gamma_d * sr + n_d  # Include gamma_d in the signal
        x_s = n_s
        A = np.outer(x_d, x_d.conj()) + np.outer(x_s, x_s.conj())
        try:
            lambda_max_A = eigvalsh(A).max()
        except np.linalg.LinAlgError:
            return np.nan  # Handle non-convergence in eigenvalue computation

        if (lambda_max_A - np.linalg.norm(x_d) ** 2) / sigma ** 2 >= gamma_th:
            PFA_count += 1
    return PFA_count / num_iterations

def main():
    gamma_th = 5
    L = 10
    num_iterations = 10000
    sigma = 1
    PT = 20  # Transmit power level
    gamma_C = 10  # Channel capacity
    gamma_Rs = [-10, 10, 20]  # Gamma_R values in dB
    rates = np.linspace(0, 8, 40)  # Rate range from 0 to 8 bpcu

    plt.figure(figsize=(8, 6))

    for gamma_R in gamma_Rs:
        results = []
        for rate in rates:
            PFA = simulate_PFA(L, num_iterations, sigma, PT, gamma_C, rate, gamma_th, gamma_R)
            results.append(PFA)

        plt.plot(rates, results, 'o-', label=f'γ_R = {gamma_R} dB')

    plt.xlabel('Rate (bpcu)', fontsize=15)
    plt.ylabel('PFA', fontsize=15)
    #plt.title('PFA vs. Rate for Different γ_R Values')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.ylim(1e-3, 1e0)  # Set limits for y-axis
    plt.xlim(0, 8)  # Set limits for x-axis
    plt.tick_params(labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-")  # Show grid lines for both major and minor ticks
    plt.show()


if __name__ == "__main__":
    print("Script started")
    main()
    print("Script finished")

