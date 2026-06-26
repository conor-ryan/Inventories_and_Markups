"""Temporary timing test: does _vfi_sweep scale with Numba thread count?
Delete after use.
"""
import sys
import time
import math
import numpy as np
import numba
sys.path.insert(0, ".")

from model_functions import (
    Parameters, solve_price_policy, precompute_demand, _vfi_sweep
)

# Parameters from SolveModel.jl line 9
params = Parameters(
    c=1.0, fc=0.0,
    mu_eta=math.log(0.05),
    sigma_eta2=0.05,
    rho_omega=0.1,
    gamma=1.0,
    delta=0.05,
    beta=0.995,
    epsilon=8.0,
    mu_nu=1.0,
    sigma_nu2=0.25,
    ns=500,
    scale=1.0,
    size=100.0,
)

print(f"Grid: ns={params.ns}, n_omega={params.q_omega}  ({params.ns * params.q_omega} cells per sweep)")
print(f"Max Numba threads available: {numba.config.NUMBA_NUM_THREADS}")
print()

# --- Build inputs for _vfi_sweep (same setup as solve_value_function) ---
p_policy = np.zeros((params.ns, params.q_omega), dtype=np.float64)
for j in range(params.q_omega):
    p_col, _ = solve_price_policy(params, params.c, params.omega_grid[j])
    p_policy[:, j] = p_col

d_table, c_table = precompute_demand(p_policy, params)
inv_h = (params.ns - 1) / (params.sgrid[-1] - params.sgrid[0])
s_lo  = float(params.sgrid[0])
v_by_omega = np.zeros((params.ns, params.q_omega), dtype=np.float64)
ev_all = params.p_omega @ v_by_omega.T

# --- Warm-up: trigger JIT compilation before any timing ---
print("Warming up JIT compilation...")
_vfi_sweep(ev_all, d_table, c_table, p_policy, params.sgrid,
           s_lo, inv_h, params.fc, params.c, params.delta, params.beta,
           params.quad_weights)
print("Done.\n")

N_REPS = 10

print(f"{'Threads':>8}  {'Total (s)':>12}  {'Per sweep (ms)':>16}  {'Speedup':>8}")
print("-" * 52)

baseline = None
for n_threads in [1, 2, 4, 8]:
    if n_threads > numba.config.NUMBA_NUM_THREADS:
        print(f"{n_threads:>8}  (exceeds pool size, skipping)")
        continue
    numba.set_num_threads(n_threads)
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        _vfi_sweep(ev_all, d_table, c_table, p_policy, params.sgrid,
                   s_lo, inv_h, params.fc, params.c, params.delta, params.beta,
                   params.quad_weights)
    elapsed = time.perf_counter() - t0
    per_sweep_ms = elapsed / N_REPS * 1000
    if baseline is None:
        baseline = elapsed
    speedup = baseline / elapsed
    print(f"{n_threads:>8}  {elapsed:>12.3f}  {per_sweep_ms:>16.1f}  {speedup:>8.2f}x")
