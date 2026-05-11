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
