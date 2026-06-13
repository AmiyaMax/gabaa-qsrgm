import numpy as np
from scipy.optimize import minimize

# Physical Constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
h = 6.62607015e-34  # Planck constant (J*s)
hbar = h / (2 * np.pi)
R = 8.314462618     # Gas constant (J/mol*K)
m_eff = 1.67e-27    # Effective mass (e.g., proton/hydrogen coordinate in kg)

def calculate_rates(T, dG_dagger, d, dE):
    """
    Calculates the combined gating rate: Classical Eyring + Quantum WKB Tunneling.
    dG_dagger: Classical activation energy (J/mol)
    d: Barrier width (meters)
    dE: Barrier height (Joules)
    """
    # 1. Classical Eyring Rate
    k_classical = (k_B * T / h) * np.exp(-dG_dagger / (R * T))
    
    # 2. Quantum Tunneling Rate (WKB)
    f_attempt = 1e9  # 1 GHz gate oscillation frequency
    wkb_exponent = -2 * (np.sqrt(2 * m_eff * dE) * d) / hbar
    k_tunnel = f_attempt * np.exp(wkb_exponent)
    
    return k_classical + k_tunnel

def negative_log_likelihood(params, temperatures, observed_rates, experimental_errors):
    """
    Objective function for MLE assuming normally distributed experimental errors.
    params: array [dG_dagger_kJ_mol, d_Angstroms, dE_kcal_mol]
    """
    dG_dagger = params[0] * 1000  # Convert kJ/mol back to J/mol
    d = params[1] * 1e-10        # Convert Angstroms to meters
    dE = params[2] * 4184         # Convert kcal/mol to Joules
    
    # Enforce strict physical boundary constraints
    if dG_dagger < 0 or d < 0 or dE < 0:
        return np.inf
        
    predicted_rates = calculate_rates(temperatures, dG_dagger, d, dE)
    
    # Gaussian log-likelihood formulation
    residuals = observed_rates - predicted_rates
    nll = 0.5 * np.sum(((residuals / experimental_errors) ** 2) + np.log(2 * np.pi * (experimental_errors ** 2)))
    return nll

# --- EXAMPLE CALIBRATION EXECUTION ---
if __name__ == "__main__":
    # Experimental Dataset (Temperatures in Kelvin, Rates in s^-1)
    temperatures = np.array([283.15, 288.15, 293.15, 298.15, 303.15, 310.15]) # 10°C to 37°C
    
    # Mock data exhibiting low-temperature upward flattening (quantum signature)
    observed_rates = np.array([0.15, 0.22, 0.38, 0.65, 1.10, 2.15]) 
    experimental_errors = np.array([0.02, 0.02, 0.03, 0.05, 0.08, 0.15])

    # Initial Guesses: [dG_dagger (kJ/mol), d (Angstroms), dE (kcal/mol)]
    initial_guess = [50.0, 4.0, 3.5]
    
    # Execute Optimization
    result = minimize(
        negative_log_likelihood, 
        initial_guess, 
        args=(temperatures, observed_rates, experimental_errors),
        method='Nelder-Mead'
    )
    
    print("=== QSRGM PARAMETRIC CALIBRATION RESULTS ===")
    if result.success:
        print(f"Optimal ΔG‡ (Classical Barrier): {result.x[0]:.2f} kJ/mol")
        print(f"Optimal Gate Barrier Width (d): {result.x[1]:.2f} Ångströms")
        print(f"Optimal Tunneling Barrier Energy (ΔE): {result.x[2]:.2f} kcal/mol")
    else:
        print("Optimization failed to converge:", result.message)