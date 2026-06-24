"""simulation_functions.py

Halton grid generation and grid-sweep moment evaluation.
Replicates halton_param_vectors and compute_moments_on_grid from
EstimationFunctions.jl.  Grid evaluation is sequential; all parallelism
is allocated to each solve_value_function call (Numba threads) and to
the SLURM job array.
"""

import os
import sys
import time
import math

import numpy as np
import pandas as pd

from model_functions import Parameters, neg_profit_check, solve_value_function
from estimation_functions import simulate_all_moments


# ---------------------------------------------------------------------------
# Halton low-discrepancy sequence
# ---------------------------------------------------------------------------

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def _radical_inverse(n, base):
    x = 0.0
    inv_base = 1.0 / base
    f = inv_base
    while n > 0:
        digit = n % base
        x += digit * f
        n = n // base
        f *= inv_base
    return x


def halton_param_vectors(bounds, n_points, seed=None):
    """Generate n_points parameter vectors using a randomized Halton sequence.

    Parameters
    ----------
    bounds   : list of (lo, hi) tuples, one per dimension
    n_points : int
    seed     : int or None

    Returns
    -------
    list of length-d numpy arrays
    """
    d = len(bounds)
    if d > len(_PRIMES):
        raise ValueError(f"Increase _PRIMES for dimension {d}")

    rng = np.random.default_rng(seed)
    lowers = np.array([b[0] for b in bounds], dtype=np.float64)
    spans  = np.array([b[1] - b[0] for b in bounds], dtype=np.float64)

    start_index = int(rng.integers(1, 100_001))
    shifts      = rng.uniform(0.0, 1.0, size=d)

    points = []
    for i in range(n_points):
        n = start_index + i
        x = np.empty(d)
        for j in range(d):
            u = _radical_inverse(n, _PRIMES[j])
            u_shift = math.fmod(u + shifts[j], 1.0)
            x[j] = lowers[j] + spans[j] * u_shift
        points.append(x)
    return points


# ---------------------------------------------------------------------------
# Grid-sweep moment evaluation  (sequential)
# ---------------------------------------------------------------------------

def compute_moments_on_grid(
    param_vectors,
    n_firms=500,
    n_years=20,
    seed=212311,
    output_path="moments.csv",
    max_value_iterations=500,
    grid_size=200,
    scale=1.0,
    size=100.0,
    solve_tol=1e-2,
):
    """Evaluate all 7 estimation moments over a list of parameter vectors.

    Replicates EstimationFunctions.jl::compute_moments_on_grid.
    Evaluation is sequential; inner parallelism comes from Numba threads
    inside solve_value_function.  Outer parallelism is via SLURM job arrays.

    Parameters
    ----------
    param_vectors          : list of length-7 arrays [γ, μη, ση2, ρω, σν2, ε, δ]
    n_firms                : int
    n_years                : int
    seed                   : int — fixed across all grid points
    output_path            : str
    max_value_iterations   : int
    grid_size              : int  — ns passed to Parameters
    scale                  : float
    size                   : float
    solve_tol              : float

    Returns
    -------
    pandas.DataFrame  (also written to output_path)
    """
    n_total = len(param_vectors)

    # Output column arrays
    out_gamma   = np.empty(n_total)
    out_mu_eta  = np.empty(n_total)
    out_sig2    = np.empty(n_total)
    out_rho     = np.empty(n_total)
    out_snu2    = np.empty(n_total)
    out_eps     = np.empty(n_total)
    out_delta   = np.empty(n_total)

    out_avg_isr          = np.full(n_total, np.nan)
    out_var_log1p_isr    = np.full(n_total, np.nan)
    out_avg_gm           = np.full(n_total, np.nan)
    out_gamma_ols        = np.full(n_total, np.nan)
    out_rho_ar1          = np.full(n_total, np.nan)
    out_sig_eta2         = np.full(n_total, np.nan)
    out_avg_opex_sales   = np.full(n_total, np.nan)
    out_failed                   = np.ones(n_total, dtype=bool)
    out_inventory_above_grid     = np.zeros(n_total, dtype=bool)
    # Fail codes: 0=success, 1=neg profit, 2=no convergence, 3=other
    out_fail_code                = np.zeros(n_total, dtype=np.int8)

    report_step = max(20, n_total // 40)
    t_start = time.time()

    for idx, pv in enumerate(param_vectors):
        gamma_i, mu_eta_i, sig2_i, rho_i, snu2_i, eps_i, delta_i = float(pv[0]), float(pv[1]), float(pv[2]), float(pv[3]), float(pv[4]), float(pv[5]), float(pv[6])

        out_gamma[idx]  = gamma_i
        out_mu_eta[idx] = mu_eta_i
        out_sig2[idx]   = sig2_i
        out_rho[idx]    = rho_i
        out_snu2[idx]   = snu2_i
        out_eps[idx]    = eps_i
        out_delta[idx]  = delta_i

        try:
            params_i = Parameters(
                mu_eta=mu_eta_i,
                sigma_eta2=sig2_i,
                rho_omega=rho_i,
                gamma=gamma_i,
                delta=delta_i,
                epsilon=eps_i,
                sigma_nu2=snu2_i,
                ns=grid_size,
                scale=scale,
                size=size,
            )

            if neg_profit_check(
                params_i.mu_log_nu, params_i.sigma_log_nu,
                params_i.epsilon, params_i.gamma,
                params_i.c, max(params_i.omega_grid),
            ):
                out_fail_code[idx] = 1
                raise RuntimeError("FAIL_NEGATIVE_PROFIT")

            sol = solve_value_function(
                params_i,
                tol=solve_tol,
                maxiter=max_value_iterations,
                conv="policy",
            )
            if not sol["converged"]:
                out_fail_code[idx] = 2
                raise RuntimeError("FAIL_NO_CONVERGENCE")

            row_seed = seed + idx
            m = simulate_all_moments(
                params_i,
                sol["p_policy"],
                sol["n_policy"],
                n_firms,
                n_years,
                row_seed,
            )

            out_avg_isr[idx]        = m["avg_isr"]
            out_var_log1p_isr[idx]  = m["var_log1p_isr"]
            out_avg_gm[idx]         = m["avg_gross_margin"]
            out_gamma_ols[idx]      = m["γ_OLS"]
            out_rho_ar1[idx]        = m["ρ_ω"]
            out_sig_eta2[idx]       = m["σ_η2"]
            out_avg_opex_sales[idx]      = m["avg_opex_sales"]
            out_inventory_above_grid[idx] = bool(m["any_inventory_above_grid"])
            out_failed[idx]               = False

        except Exception as e:
            msg = str(e)
            if "FAIL_NEGATIVE_PROFIT" not in msg and "FAIL_NO_CONVERGENCE" not in msg:
                out_fail_code[idx] = 3
                print(f"Grid point {idx+1}/{n_total} failed with unexpected error: {msg}")
        # Progress bar
        done = idx + 1
        if done % report_step == 0 or done == n_total:
            elapsed = time.time() - t_start
            pct = done / n_total
            eta = elapsed / pct * (1.0 - pct) if pct < 1.0 else 0.0
            bar_len = 40
            filled = round(bar_len * pct)
            bar = "=" * max(0, filled - 1) + ">" + " " * (bar_len - filled)
            def fmt(s):
                h, r = divmod(int(s), 3600)
                m2, s2 = divmod(r, 60)
                return f"{h:02d}:{m2:02d}:{s2:02d}"
            print(
                f"\r  [{bar}] {100.0*pct:3.0f}%  {done}/{n_total}"
                f"  (elapsed {fmt(elapsed)}, ETA {fmt(eta)})   ",
                end="", flush=True,
            )
    print()

    df_out = pd.DataFrame({
        "gamma":              out_gamma,
        "mu_eta":             out_mu_eta,
        "sigma_eta2":         out_sig2,
        "rho_omega":          out_rho,
        "sigma_nu2":          out_snu2,
        "epsilon":            out_eps,
        "delta":              out_delta,
        "avg_isr":            out_avg_isr,
        "var_log1p_isr":      out_var_log1p_isr,
        "avg_gross_margin":   out_avg_gm,
        "gamma_OLS":          out_gamma_ols,
        "rho_omega_ar1":      out_rho_ar1,
        "sigma_eta2_ar1":     out_sig_eta2,
        "avg_opex_sales":              out_avg_opex_sales,
        "any_inventory_above_grid":   out_inventory_above_grid,
        "failed":                     out_failed,
        "fail_code":                  out_fail_code,
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_out.to_csv(output_path, index=False)

    n_ok = int((~out_failed).sum())
    n_failed = n_total - n_ok
    print(f"\nGrid search complete.  {n_ok} / {n_total} points succeeded.  Results → {output_path}")
    if n_failed > 0:
        fc = out_fail_code[out_failed]
        print(
            f"Failure breakdown (among {n_failed} failed): "
            f"neg_profit={int((fc==1).sum())}, "
            f"no_convergence={int((fc==2).sum())}, "
            f"other={int((fc==3).sum())}"
        )
    frac_above_grid = out_inventory_above_grid[~out_failed].mean() if n_ok > 0 else float("nan")
    print(f"Fraction of successful runs with any simulated inventory above Smax: {frac_above_grid:.6f}")

    return df_out
