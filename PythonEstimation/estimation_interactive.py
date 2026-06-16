"""
Interactive estimation workflow. Run blocks sequentially in a Python terminal.
Mirrors estimation_main.py but without argparse — edit the config block below.
"""

# %% Imports
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model_functions import Parameters, solve_value_function
from estimation_functions import (
    simulate_all_moments,
    compute_ii_asymptotic_variance,
    compute_simulation_variance,
    select_best_grid_start,
)
from global_max import tiktak


# %% Config — edit these before running
DATASET_ID = "004"
DATA_DIR   = Path(__file__).resolve().parents[2] / "SimulatedData"

MOMENT_NAMES = ["avg_isr", "var_log1p_isr", "avg_gross_margin",
                "γ_OLS", "ρ_ω", "σ_η2", "avg_opex_sales"]
PARAM_NAMES  = ["gamma", "mu_eta", "sigma_eta2", "rho_omega",
                "sigma_nu2", "epsilon", "delta"]

PARAMS_BASE = Parameters(
    c=1.0, fc=0.0, beta=0.995, mu_nu=1.0,
    sigma_nu2=0.15, ns=200, scale=1.0, size=100.0,
    mu_eta=math.log(0.01), sigma_eta2=0.05,
    rho_omega=0.1, gamma=0.9, delta=0.01, epsilon=8.0,
)

output_path  = DATA_DIR / f"estimates_id_{DATASET_ID}.csv"
moments_path = DATA_DIR / f"implied_moments_id_{DATASET_ID}.csv"

print(f"=== Estimation: dataset {DATASET_ID} ===")
print(f"  data_dir : {DATA_DIR}")
print(f"  output   : {output_path}")


# %% 1. Load inputs
df_target = pd.read_csv(DATA_DIR / f"target_moments_id_{DATASET_ID}.csv")
target_moments = dict(zip(df_target["moment"], df_target["value"].astype(float)))

df_vcov = pd.read_csv(DATA_DIR / f"target_moment_vcov_id_{DATASET_ID}.csv", index_col=0)
vcov = df_vcov.loc[MOMENT_NAMES, MOMENT_NAMES].to_numpy(dtype=np.float64)
W    = np.linalg.inv(vcov)

df_grid = pd.read_csv(DATA_DIR / "moments.csv")

print("\nTarget moments:")
for k in MOMENT_NAMES:
    print(f"  {k:22s} = {target_moments[k]:.6f}")


# %% 2. Weighting matrix: combine data variance and simulation variance
print("\nStep 2: computing combined weighting matrix ...")
best_grid = select_best_grid_start(df_grid, target_moments, W)
print(f"  Best grid point  (obj = {best_grid['obj_value']:.6f})")

params_grid = Parameters(
    c=PARAMS_BASE.c, fc=PARAMS_BASE.fc,
    beta=PARAMS_BASE.beta, mu_nu=PARAMS_BASE.mu_nu,
    ns=PARAMS_BASE.ns, size=PARAMS_BASE.size,
    gamma=best_grid["gamma"],         mu_eta=best_grid["mu_eta"],
    sigma_eta2=best_grid["sigma_eta2"], rho_omega=best_grid["rho_omega"],
    sigma_nu2=best_grid["sigma_nu2"],  epsilon=best_grid["epsilon"],
    delta=best_grid["delta"],
)
sol_grid = solve_value_function(params_grid)
vcov_sim = compute_simulation_variance(
    params_grid, sol_grid["p_policy"], sol_grid["n_policy"],
    n_firms=500, n_years=20, n_reps=500, seed=212311,
    moment_keys=MOMENT_NAMES, verbose=False,
)

W = np.linalg.inv(vcov + vcov_sim)
print("  Combined weighting matrix computed.")


# %% 3. Estimate
print("\nRunning Tik-Tak estimation ...")
t0 = time.perf_counter()
ii_result = tiktak(
    df_grid,
    target_moments,
    W,
    params_base=PARAMS_BASE,
    n_firms=500,
    n_years=20,
    seed=212311,
    max_iter=1000,
    n_iterations=20,
    alpha_min=0.1,
    verbose=True,
    method="nelder-mead",
    tol=1e-2,
    tol_final=1e-4,
)
print(f"\n  Estimation wall time : {time.perf_counter() - t0:.1f}s")
print(f"  Objective            : {ii_result['obj_value']:.8f}")
print("  Tik-Tak history:")
for rec in ii_result["tiktak_history"]:
    print(f"    iter {str(rec[0]):>5s}  obj={rec[1]:.8f}  alpha={rec[2]:.3f}")


# %% 4. Standard errors
params_hat = Parameters(
    c=PARAMS_BASE.c, fc=PARAMS_BASE.fc,
    beta=PARAMS_BASE.beta, mu_nu=PARAMS_BASE.mu_nu,
    ns=PARAMS_BASE.ns, size=PARAMS_BASE.size,
    gamma=ii_result["gamma"],         mu_eta=ii_result["mu_eta"],
    sigma_eta2=ii_result["sigma_eta2"], rho_omega=ii_result["rho_omega"],
    sigma_nu2=ii_result["sigma_nu2"],  epsilon=ii_result["epsilon"],
    delta=ii_result["delta"],
)

t0 = time.perf_counter()
se_results = compute_ii_asymptotic_variance(
    params_hat, W,
    n_firms=5000,
    n_years=20,
    seed=212311,
    verbose=True,
)
print(f"\n  SE wall time : {time.perf_counter() - t0:.1f}s")


# %% 5. Save estimates
rows = []
for name, se in zip(PARAM_NAMES, se_results["se"]):
    rows.append({"parameter": name, "estimate": ii_result[name], "std_error": se})
rows.append({"parameter": "obj_value", "estimate": ii_result["obj_value"], "std_error": float("nan")})

df_out = pd.DataFrame(rows)
df_out.to_csv(output_path, index=False)
print(f"Saved estimates to {output_path}")


# %% 6. Implied moments at estimated parameters
sol_hat = solve_value_function(params_hat)
implied_moments = simulate_all_moments(
    params_hat, sol_hat["p_policy"], sol_hat["n_policy"],
    n_firms=500, n_years=20, seed=212311,
)

df_moments = pd.DataFrame([
    {"moment": k, "implied": implied_moments[k], "target": target_moments[k]}
    for k in MOMENT_NAMES
])
df_moments.to_csv(moments_path, index=False)
print(f"Saved implied moments to {moments_path}")

print(f"\n{'moment':22s}  {'implied':>10s}  {'target':>10s}  {'diff':>10s}")
print("-" * 58)
for k in MOMENT_NAMES:
    imp = implied_moments[k]
    tgt = target_moments[k]
    print(f"  {k:20s}  {imp:10.6f}  {tgt:10.6f}  {imp - tgt:+10.6f}")


# %% Summary table
print(f"\n{'parameter':14s}  {'estimate':>12s}  {'std_error':>12s}")
print("-" * 42)
for name, se in zip(PARAM_NAMES, se_results["se"]):
    print(f"  {name:12s}  {ii_result[name]:12.6f}  {se:12.6f}")
