"""Thread-scaling benchmark: solve_price_policy vs VFI sweep vs total solve_value_function.

Run standalone or via run_test_thread_scaling.slurm.
Reads SLURM_CPUS_PER_TASK to set the Numba thread-pool size, then sweeps
thread counts from 1 up to that maximum in powers of 2.

Usage:
    python test_thread_scaling.py               # local, uses os.cpu_count()
    sbatch run_test_thread_scaling.slurm        # server
"""
import os
import sys
import time
import math

import numba
_max_threads = int(
    os.environ.get("SLURM_CPUS_PER_TASK") or
    os.environ.get("NCPUS") or
    os.cpu_count() or 1
)
numba.set_num_threads(_max_threads)

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from model_functions import (
    Parameters, solve_price_policy, precompute_demand,
    _vfi_sweep, solve_value_function,
)

# ---------------------------------------------------------------------------
# Parameters from SolveModel.jl line 9
# ---------------------------------------------------------------------------
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

print(f"Grid: ns={params.ns}, n_omega={params.q_omega}  "
      f"({params.ns * params.q_omega} cells per sweep)")
print(f"Numba thread-pool size: {numba.config.NUMBA_NUM_THREADS}")
print()

# ---------------------------------------------------------------------------
# 1. Time solve_price_policy (sequential — same for every thread count)
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
for j in range(params.q_omega):
    solve_price_policy(params, params.c, params.omega_grid[j])
t_price = time.perf_counter() - t0
print(f"solve_price_policy ({params.q_omega} omega states): {t_price*1000:.1f} ms"
      f"  [sequential — constant across thread counts]\n")

# ---------------------------------------------------------------------------
# 2. Warm up JIT at max threads before any timing
# ---------------------------------------------------------------------------
p_policy = np.zeros((params.ns, params.q_omega), dtype=np.float64)
for j in range(params.q_omega):
    p_col, _ = solve_price_policy(params, params.c, params.omega_grid[j])
    p_policy[:, j] = p_col
d_table, c_table = precompute_demand(p_policy, params)
inv_h = (params.ns - 1) / (params.sgrid[-1] - params.sgrid[0])
s_lo  = float(params.sgrid[0])
v_by_omega = np.zeros((params.ns, params.q_omega), dtype=np.float64)
ev_all = params.p_omega @ v_by_omega.T

print("Warming up JIT compilation...")
_vfi_sweep(ev_all, d_table, c_table, p_policy, params.sgrid,
           s_lo, inv_h, params.fc, params.c, params.delta, params.beta,
           params.quad_weights)
print("Done.\n")

# ---------------------------------------------------------------------------
# 3. Sweep thread counts
# ---------------------------------------------------------------------------
N_SWEEP_REPS = 10   # reps for per-sweep timing
TOL  = 1e-2
MAXITER = 500

thread_counts = []
n = 1
while n <= _max_threads:
    thread_counts.append(n)
    n *= 2

hdr = (f"{'Threads':>8}  {'price_policy (ms)':>18}  "
       f"{'per sweep (ms)':>15}  {'VFI iters':>10}  "
       f"{'total svf (ms)':>15}")
print(hdr)
print("-" * len(hdr))

for n_threads in thread_counts:
    numba.set_num_threads(n_threads)

    # --- per-sweep timing (isolated _vfi_sweep) ---
    t0 = time.perf_counter()
    for _ in range(N_SWEEP_REPS):
        _vfi_sweep(ev_all, d_table, c_table, p_policy, params.sgrid,
                   s_lo, inv_h, params.fc, params.c, params.delta, params.beta,
                   params.quad_weights)
    per_sweep_ms = (time.perf_counter() - t0) / N_SWEEP_REPS * 1000

    # --- total solve_value_function ---
    t0 = time.perf_counter()
    sol = solve_value_function(params, tol=TOL, maxiter=MAXITER, conv="policy")
    t_svf_ms = (time.perf_counter() - t0) * 1000

    n_iters = sol["iterations"]

    print(f"{n_threads:>8}  {t_price*1000:>18.1f}  "
          f"{per_sweep_ms:>15.1f}  {n_iters:>10}  "
          f"{t_svf_ms:>15.1f}")
