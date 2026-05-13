"""
Interactive workflow for Python replication of Julia solve_value_function.

Copy-paste blocks into a Python terminal in order.
"""

# %% Imports and setup
from pathlib import Path
import subprocess

import numpy as np

from model_functions import Parameters, solve_value_function


ROOT = Path.cwd()  # Assumes you're running from PythonEstimation directory
REFERENCE_DIR = ROOT / "parity_reference"


def run_julia_export():
    script = ROOT / "export_julia_reference.jl"
    subprocess.run(["julia", str(script)], check=True)


def _load_vector(name):
    arr = np.loadtxt(REFERENCE_DIR / f"{name}.csv", delimiter=",")
    return np.atleast_1d(np.asarray(arr, dtype=np.float64))


def _load_matrix(name):
    arr = np.loadtxt(REFERENCE_DIR / f"{name}.csv", delimiter=",")
    return np.asarray(arr, dtype=np.float64)


def _max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


def _max_rel_diff(a, b):
    scale = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / scale)


def _print_comparison(name, py_arr, jl_arr):
    print(
        f"{name}: shape={py_arr.shape}, max_abs_diff={_max_abs_diff(py_arr, jl_arr):.6e}, max_rel_diff={_max_rel_diff(py_arr, jl_arr):.6e}"
    )


# %% Target parameterization from SolveModel.jl line 6
ns = 200
params = Parameters(
    c=1.0,
    fc=0.0,
    mu_eta=float(__import__("math").log(0.01)),
    sigma_eta2=0.05,
    rho_omega=0.1,
    gamma=0.9,
    delta=0.01,
    beta=0.95,
    epsilon=8.0,
    mu_nu=1.0,
    sigma_nu2=0.15,
    ns=ns,
    scale=1.0,
    size=100.0,
)


# %% Solve VFI — first call (includes Numba JIT compile time)
import time

t0 = time.perf_counter()
result = solve_value_function(params, tol=1e-4, maxiter=1000)
t1 = time.perf_counter()
print(f"First call (includes JIT compile): {t1 - t0:.2f}s")


# %% Solve VFI — 10 timed runs (compiled cache, true runtime)
N_RUNS = 10
times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    result = solve_value_function(params, tol=1e-4, maxiter=1000)
    times.append(time.perf_counter() - t0)

times = np.array(times)
n_iter = result["iterations"]
print(f"Timed runs (n={N_RUNS}):")
print(f"  mean:   {times.mean():.3f}s  |  std: {times.std():.3f}s")
print(f"  min:    {times.min():.3f}s  |  max: {times.max():.3f}s")
print(f"  median: {np.median(times):.3f}s")
print(f"  per VFI iteration (mean): {times.mean() / n_iter:.4f}s")
print(f"  per (s, ω) state  (mean): {times.mean() / n_iter / params.ns / params.q_omega * 1e6:.2f} µs")


# %% Quick diagnostics
print("Converged:", result["converged"])
print("Iterations:", result["iterations"])
print("Final sup norm diff:", result["final_diff"])
print("V[:5]:", result["v"][:5])
print("n_policy[0, :]:", result["n_policy"][0, :])
print("p_policy[0, :]:", result["p_policy"][0, :])


# %% Optional: export Julia reference outputs from Python
# Requires Julia on PATH and export_julia_reference.jl in this folder.
# run_julia_export()


# %% Load Julia reference outputs (after export)
jl = {
    "sgrid": _load_vector("sgrid"),
    "omega_grid": _load_vector("omega_grid"),
    "pi_omega": _load_vector("pi_omega"),
    "p_omega": _load_matrix("p_omega"),
    "v": _load_vector("v"),
    "n_policy": _load_matrix("n_policy"),
    "p_policy": _load_matrix("p_policy"),
    "v_by_omega": _load_matrix("v_by_omega"),
    "converged": int(_load_vector("converged")[0]),
}


# %% Parity comparisons: Python vs Julia
_print_comparison("sgrid", params.sgrid, jl["sgrid"])
_print_comparison("omega_grid", params.omega_grid, jl["omega_grid"])
_print_comparison("pi_omega", params.pi_omega, jl["pi_omega"])
_print_comparison("p_omega", params.p_omega, jl["p_omega"])
_print_comparison("v", result["v"], jl["v"])
_print_comparison("n_policy", result["n_policy"], jl["n_policy"])
_print_comparison("p_policy", result["p_policy"], jl["p_policy"])
_print_comparison("v_by_omega", result["v_by_omega"], jl["v_by_omega"])

py_conv = 1 if result["converged"] else 0
print(f"converged: python={py_conv}, julia={jl['converged']}")


# ==========================================================================
# STAGE 2 — Estimation
# ==========================================================================

# %% Stage 2 imports
from estimation_functions import (
    compute_ii_jacobian,
    compute_ii_asymptotic_variance,
    compute_simulation_variance,
    simulate_firm,
    simulate_all_moments,
    select_best_grid_start,
    estimate_params_ii_full,
)
from global_max import tiktak
import pandas as pd


# %% Load target moments and weighting matrix
# Edit paths to match where your data files are stored.
SIM_DATA_DIR = ROOT.parent.parent / "SimulatedData"
target_moments_path = SIM_DATA_DIR / "target_moments_id_004.csv"
target_vcov_path    = SIM_DATA_DIR / "target_moment_vcov_id_004.csv"
grid_path           = SIM_DATA_DIR / "moments.csv"

MOMENT_NAMES = ["avg_isr", "var_log1p_isr", "avg_gross_margin",
                "γ_OLS", "ρ_ω", "σ_η2", "avg_opex_sales"]

df_target  = pd.read_csv(target_moments_path)
target_moments = dict(zip(df_target["moment"], df_target["value"].astype(float)))

df_vcov = pd.read_csv(target_vcov_path, index_col=0)
vcov    = df_vcov.loc[MOMENT_NAMES, MOMENT_NAMES].to_numpy(dtype=np.float64)
W       = np.linalg.inv(vcov)


print("Target moments:")
for k in MOMENT_NAMES:
    print(f"  {k:22s} = {target_moments[k]:.6f}")


# %% Select warm-start from best grid point
true_params_path = SIM_DATA_DIR / "true_parameters_id_004.csv"
df_true = pd.read_csv(true_params_path)
true_vals = {
    "gamma":      float(df_true["γ"].iloc[0]),
    "mu_eta":     float(df_true["μη"].iloc[0]),
    "sigma_eta2": float(df_true["ση2"].iloc[0]),
    "rho_omega":  float(df_true["ρ_ω"].iloc[0]),
    "sigma_nu2":  float(df_true["σν2"].iloc[0]),
    "epsilon":    float(df_true["ϵ"].iloc[0]),
    "delta":      float(df_true["δ"].iloc[0]),
}

# %% Simulation covariance and asymptotic variance at true parameters
params_true = Parameters(
    c=params.c, fc=params.fc, beta=params.beta, ns=params.ns, size=params.size,
    mu_nu=params.mu_nu,
    gamma=true_vals["gamma"],         mu_eta=true_vals["mu_eta"],
    sigma_eta2=true_vals["sigma_eta2"], rho_omega=true_vals["rho_omega"],
    sigma_nu2=true_vals["sigma_nu2"],  epsilon=true_vals["epsilon"],
    delta=true_vals["delta"],
)
sol_true = solve_value_function(params_true)

vcov_sim_true, draws_sim_true = compute_simulation_variance(
    params_true, sol_true["p_policy"], sol_true["n_policy"],
    n_firms=500, n_years=20, n_reps=50, seed=212311,
    moment_keys=MOMENT_NAMES,
)


W_sim= np.linalg.inv(vcov_sim_true + vcov)

avar_true = compute_ii_asymptotic_variance(
    params_true, W_sim_true, n_firms=5000, n_years=20, seed=0,
)
print("\nSimulation vcov and avar computed at true parameters.")


df_grid = pd.read_csv(grid_path)
start   = select_best_grid_start(df_grid, target_moments, W)
init_guess = [start["gamma"], start["mu_eta"], start["sigma_eta2"],
              start["rho_omega"], start["sigma_nu2"], start["epsilon"],
              start["delta"]]
print(f"\nWarm-start: BEST GRID  (obj = {start['obj_value']:.6f})")

# --- Diagnostic: re-solve at best grid point and compare moments ---
# Checks whether the grid moments are reproducible.  Large discrepancies
# indicate simulation noise, a changed seed, or a grid/model mismatch.
params_grid = Parameters(
    c=params.c, fc=params.fc, beta=params.beta, ns=params.ns, size=params.size,
    gamma=init_guess[0],      mu_eta=init_guess[1],     sigma_eta2=init_guess[2],
    rho_omega=init_guess[3],  sigma_nu2=init_guess[4],  epsilon=init_guess[5],
    delta=init_guess[6],
)
sol_grid     = solve_value_function(params_grid)
moments_grid = simulate_all_moments(
    params_grid, sol_grid["p_policy"], sol_grid["n_policy"],
    n_firms=500, n_years=20, seed=212311,
)
grid_row = df_grid.loc[start["row_index"]]
print(f"\nDiagnostic: re-solved moments vs grid CSV at best grid point")
print(f"{'moment':22s}  {'re-solved':>10s}  {'grid CSV':>10s}  {'diff':>10s}")
print("-" * 60)
for k in MOMENT_NAMES:
    resol  = moments_grid[k]
    stored = float(grid_row[k])
    print(f"  {k:20s}  {resol:10.6f}  {stored:10.6f}  {resol - stored:+10.6f}")


# %% Check: simulated moments at TRUE parameters vs target moments
# Verifies that the simulator can reproduce the data moments when given the
# true DGP parameters.  Large discrepancies indicate a simulator mismatch.
params_true = Parameters(
    c=params.c, fc=params.fc, beta=params.beta, ns=params.ns, size=params.size,
    mu_eta=true_vals["mu_eta"],     sigma_eta2=true_vals["sigma_eta2"],
    rho_omega=true_vals["rho_omega"], gamma=true_vals["gamma"],
    delta=true_vals["delta"],         epsilon=true_vals["epsilon"],
    sigma_nu2=true_vals["sigma_nu2"],
)
sol_true = solve_value_function(params_true)

t0 = time.perf_counter()
moments_true = simulate_all_moments(
    params_true, sol_true["p_policy"], sol_true["n_policy"],
    n_firms=500, n_years=20, seed=212311,
)
print(f"simulate_all_moments @ true params (500 firms, 20 years): {time.perf_counter()-t0:.2f}s")
print(f"\n{'moment':22s}  {'simulated':>10s}  {'target':>10s}  {'diff':>10s}")
print("-" * 58)
for k in MOMENT_NAMES:
    sim = moments_true[k]
    tgt = target_moments[k]
    print(f"  {k:20s}  {sim:10.6f}  {tgt:10.6f}  {sim - tgt:+10.6f}")


# %% Run simulate_all_moments at start guess (smoke test before estimation)
# init_guess = [gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta]
params_start = Parameters(
    c=params.c, fc=params.fc, beta=params.beta, ns=params.ns, size=params.size,
    gamma=init_guess[0],      mu_eta=init_guess[1],     sigma_eta2=init_guess[2],
    rho_omega=init_guess[3],  sigma_nu2=init_guess[4],  epsilon=init_guess[5],
    delta=init_guess[6],
)
sol_start = solve_value_function(params_start)

t0 = time.perf_counter()
moments_start = simulate_all_moments(
    params_start, sol_start["p_policy"], sol_start["n_policy"],
    n_firms=500, n_years=20, seed=212311,
)
print(f"\nsimulate_all_moments @ start guess (500 firms, 20 years): {time.perf_counter()-t0:.2f}s")
print("Simulated moments at start guess:")
for k in MOMENT_NAMES:
    print(f"  {k:22s} = {moments_start[k]:.6f}  (target {target_moments[k]:.6f})")


# %% Run Tik-Tak estimation
t0 = time.perf_counter()
ii_result = tiktak(
    df_grid,
    target_moments,
    W,
    params_base=params,
    n_firms=500,
    n_years=20,
    seed=212311,
    max_iter=1000,
    n_iterations=10,
    alpha_min=0.1,
    verbose=True,
)
print(f"\nTik-Tak wall time: {time.perf_counter()-t0:.1f}s")


# %% Print final estimates
print("\nEstimated parameters:")
for k in ["gamma","mu_eta","sigma_eta2","rho_omega","sigma_nu2","epsilon","delta"]:
    print(f"  {k:12s} = {ii_result[k]:.6f}")
print(f"  obj_value    = {ii_result['obj_value']:.8f}")
print("\nTik-Tak history (iter, obj, alpha):")
for rec in ii_result["tiktak_history"]:
    print(f"  iter {rec[0]}  obj={rec[1]:.8f}  alpha={rec[2]:.3f}")


# %% Standard errors — asymptotic variance
from estimation_functions import compute_ii_asymptotic_variance

# Build Parameters at the estimated point
params_hat = Parameters(
    c=params.c, fc=params.fc, beta=params.beta, ns=params.ns, size=params.size,
    mu_eta=ii_result["mu_eta"],     sigma_eta2=ii_result["sigma_eta2"],
    rho_omega=ii_result["rho_omega"], gamma=ii_result["gamma"],
    delta=ii_result["delta"],         epsilon=ii_result["epsilon"],
    sigma_nu2=ii_result["sigma_nu2"],
)

# sample_size = number of firms in the actual data used to compute target moments
SAMPLE_SIZE = 1  # replace with actual N if known

t0 = time.perf_counter()
se_results = compute_ii_asymptotic_variance(
    params_hat, W,
    n_firms=5000,
    n_years=20,
    seed=212311,
    sample_size=SAMPLE_SIZE,
    verbose=True,
)
print(f"\nJacobian wall time: {time.perf_counter()-t0:.1f}s")


# %% Print results table (estimates + SEs + true values)
PARAM_NAMES = ["gamma", "mu_eta", "sigma_eta2", "rho_omega", "sigma_nu2", "epsilon", "delta"]
# true_vals already loaded in the warm-start block above

print("\n{:14s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
    "parameter", "true", "estimate", "std_error", "bias/SE"))
print("-" * 70)
for name, se in zip(PARAM_NAMES, se_results["se"]):
    est  = ii_result[name]
    true = true_vals[name]
    bias_se = (est - true) / se if se > 0 else float("nan")
    print(f"  {name:12s}  {true:12.6f}  {est:12.6f}  {se:12.6f}  {bias_se:12.3f}")

# Save to CSV
df_results = pd.DataFrame({
    "parameter": PARAM_NAMES,
    "estimate":  [ii_result[k] for k in PARAM_NAMES],
    "std_error": list(se_results["se"]),
})
results_path = SIM_DATA_DIR / "estimates_id_004.csv"
df_results.to_csv(results_path, index=False)
print(f"\nSaved to {results_path}")


