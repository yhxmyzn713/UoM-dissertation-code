import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from scipy.special import gamma as gamma_func, comb, hyp1f1

def main():
    PT = 20  
    L = 10  
    gamma_values = np.linspace(0, 10, 25)  
    num_iterations = 10000  
    sigma = 1  

    D_SNR_dBs = np.array([-10, 10, 20])  # D-SNR (dB)
    alpha_R = 2 * 10 ** (D_SNR_dBs / 10)  # Convert dB to linear scale for theoretical calculations

    PFA_simulated = {snr: np.zeros(len(gamma_values)) for snr in D_SNR_dBs}
    pfa_theoretical = []

    # Simulated PFA vs Threshold γ
    for idx, D_SNR_dB in enumerate(D_SNR_dBs):
        D_SNR = 10 ** (D_SNR_dB / 10)
        for i, gamma in enumerate(gamma_values):
            PFA_count = simulate_PFA(L, num_iterations, sigma, D_SNR, gamma)
            PFA_simulated[D_SNR_dB][i] = PFA_count / num_iterations

    # Calculate theoretical PFA
    for alpha in alpha_R:
        pfa = []
        for gamma in gamma_values:
            term1 = np.exp(-gamma)
            term2 = (2 * gamma * np.exp(-(gamma + alpha / 2))) / (2 ** L * gamma_func(L))

            sum_term1 = 0
            for k in range(L - 1):
                for p in range(k + 1):
                    sum_term1 += (comb(k, p) * 2 ** (L - 1) * gamma ** (k - p) *
                                  gamma_func(L + p - k - 1) *
                                  hyp1f1(L + p - k - 1, L, alpha / 2))

            sum_term2 = 0
            for k in range(L - 1):
                for l in range(k + 1):
                    for p in range(k + 1):
                        sum_term2 += (comb(k, p) * 2 ** (k - p - l) * gamma ** (k - p) /
                                      gamma_func(l + 1) *
                                      gamma_func(L + l + p - k - 1) *
                                      hyp1f1(L + l + p - k - 1, L, alpha / 4))

            pfa_value = term1 + term2 * (sum_term1 - sum_term2)
            pfa.append(pfa_value)
        pfa_theoretical.append(pfa)

    # Calculate High D-SNR approx PFA_high
    PFA_high = np.exp(-gamma_values)

    # Plotting results
    plot_results(gamma_values, PFA_simulated, pfa_theoretical, D_SNR_dBs, PFA_high)
    plt.show()

def simulate_PFA(L, num_iterations, sigma, Pr, gamma_th):
    PFA_count = 0
    for _ in range(num_iterations):
        n_d, n_s = generate_noise(L, sigma)
        x_d = generate_signal(L, Pr) + n_d
        x_s = n_s
        A = np.outer(x_d, x_d.conj()) + np.outer(x_s, x_s.conj())
        lambda_max_A = max(eigvals(A).real)
        if (lambda_max_A - np.linalg.norm(x_d) ** 2) / sigma >= gamma_th:
            PFA_count += 1
    return PFA_count

def generate_noise(L, sigma):
    return (np.sqrt(sigma / 2) * (np.random.randn(L) + 1j * np.random.randn(L)),
            np.sqrt(sigma / 2) * (np.random.randn(L) + 1j * np.random.randn(L)))

def generate_signal(L, power):
    signal = np.full((L,), np.sqrt(power / L), dtype=complex)
    actual_power = np.sum(np.abs(signal)**2)
    normalization_factor = np.sqrt(power / actual_power)
    normalized_signal = signal * normalization_factor
    return normalized_signal


def plot_results(gamma_values, PFA_simulated, pfa_theoretical, D_SNR_dBs, PFA_high):
    plt.figure(figsize=(8, 6))

    for idx, D_SNR_dB in enumerate(D_SNR_dBs):
        # Plot simulated PFA using dots
        plt.plot(gamma_values, PFA_simulated[D_SNR_dB], 'o', label=f'Sim.D-SNR={D_SNR_dB} dB')
        # Plot theoretical PFA using lines
        plt.plot(gamma_values, pfa_theoretical[idx], '-', label=f'Theo.D-SNR={D_SNR_dB} dB')

    plt.plot(gamma_values,PFA_high,'-->', label='High D-SNR Approx', color='black')

    # plt.title('PFA versus γ for different D-SNR values', fontsize=16)
    plt.xlabel('γ', fontsize=15)
    plt.ylabel('PFA', fontsize=15)
    plt.ylim(0, 1)
    plt.xlim(0, 10)
    plt.tick_params(labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script finished")
