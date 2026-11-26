"""
QSRGM calibration and model-validation script.
Journal-neutral reproducibility version.

Important:
- The example dataset below is SYNTHETIC and is included only to test the code.
- It must not be described as experimental calibration.
- WKB barrier energy is converted from molar units to single-particle joules
  before entering the WKB exponent.
- All transition/rate quantities remain non-negative.
- d and sqrt(DeltaE) are structurally correlated in the rectangular-barrier WKB term;
  they should not be interpreted as independently identifiable from one temperature
  series without additional constraints.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution

K_B = 1.380649e-23          # J K^-1
H = 6.62607015e-34          # J s
HBAR = H / (2.0 * np.pi)
R = 8.314462618             # J mol^-1 K^-1
N_A = 6.02214076e23         # mol^-1
M_EFF = 1.67e-27            # kg; declared effective-mass model assumption


def wkb_exponent_rectangular(d_m, delta_e_kcal_mol, m_eff=M_EFF):
    """Dimensionless WKB exponent for a rectangular effective barrier."""
    delta_e_particle = delta_e_kcal_mol * 4184.0 / N_A  # J per particle
    return 2.0 * d_m * np.sqrt(2.0 * m_eff * delta_e_particle) / HBAR


def calculate_rates(T, dG_kJ_mol, d_A, delta_e_kcal_mol,
                    attempt_frequency=1.0e9):
    """
    Combined classical Eyring + phenomenological WKB pathway.

    Parameters are intentionally phenomenological unless independently constrained.
    """
    T = np.asarray(T, dtype=float)
    dG_J_mol = dG_kJ_mol * 1000.0
    d_m = d_A * 1.0e-10

    # Classical pathway: non-negative Eyring rate.
    k_classical = (K_B * T / H) * np.exp(-dG_J_mol / (R * T))

    # Quantum pathway: rectangular effective barrier.
    gamma = wkb_exponent_rectangular(d_m, delta_e_kcal_mol)
    k_tunnel = attempt_frequency * np.exp(-gamma)

    return k_classical + k_tunnel, k_classical, k_tunnel


def nll_gaussian(params, temperatures, observed_rates, errors):
    """Negative log-likelihood for normally distributed rate errors."""
    dG, d_A, dE = params
    if dG <= 0 or d_A <= 0 or dE <= 0 or np.any(errors <= 0):
        return np.inf

    predicted, _, _ = calculate_rates(temperatures, dG, d_A, dE)
    residuals = observed_rates - predicted
    return 0.5 * np.sum(
        (residuals / errors) ** 2 +
        np.log(2.0 * np.pi * errors ** 2)
    )


def fit_model(temperatures, observed_rates, errors):
    """Bounded global search followed by local refinement."""
    bounds = [(1.0, 150.0), (0.5, 8.0), (0.1, 8.0)]

    global_result = differential_evolution(
        nll_gaussian,
        bounds=bounds,
        args=(temperatures, observed_rates, errors),
        seed=7,
        polish=True,
    )

    local_result = minimize(
        nll_gaussian,
        global_result.x,
        args=(temperatures, observed_rates, errors),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return local_result


def synthetic_dataset():
    """
    Demonstration only. These numbers are synthetic and are NOT experimental data.
    """
    T = np.array([283.15, 288.15, 293.15, 298.15, 303.15, 310.15])
    true_params = np.array([55.0, 2.2, 1.5])
    rates, _, _ = calculate_rates(T, *true_params, attempt_frequency=1.0e9)
    rng = np.random.default_rng(7)
    errors = np.maximum(0.03 * rates, 0.02)
    observed = rates + rng.normal(0.0, errors)
    return T, observed, errors, true_params


if __name__ == "__main__":
    T, observed, errors, true_params = synthetic_dataset()
    result = fit_model(T, observed, errors)

    fitted = result.x
    rates, k_classical, k_tunnel = calculate_rates(T, *fitted)

    print("QSRGM SYNTHETIC CODE-VALIDATION RUN")
    print("WARNING: dataset is synthetic; this is not experimental calibration.")
    print(f"Converged: {result.success}")
    print(f"dG^‡       = {fitted[0]:.4f} kJ/mol")
    print(f"d          = {fitted[1]:.4f} Å")
    print(f"Delta E    = {fitted[2]:.4f} kcal/mol")
    print(f"NLL        = {result.fun:.6f}")
    print(f"WKB gamma  = {wkb_exponent_rectangular(fitted[1]*1e-10, fitted[2]):.6g}")
    print(f"True demo parameters = {true_params}")
