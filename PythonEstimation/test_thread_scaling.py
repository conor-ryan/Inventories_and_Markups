"""Thread-scaling benchmark for _vfi_sweep and solve_price_policy.

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
# Set pool size to the full allocation before any JIT compilation.
_max_threads = int(
    os.environ.get("SLURM_CPUS_PER_TASK") or
    os.environ.get("NCPUS") or
    os.cpu_count() or 1
)
numba.set_num_threads(_max_threads)

import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from model_functions import (
    Parameters, solve_price_policy, precompute_demand, _vfi_sweep
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
print(f"Threads used at start:  {numba.get_num_threads()}")
print()

# ---------------------------------------------------------------------------
# Build inputs for _vfi_sweep
# ---------------------------------------------------------------------------
print("Computing price policy (sequential, for reference timing)...")
t0 = time.perf_counter()
p_policy = np.zeros((params.ns, params.q_omega), dtype=np.float64)
for j in range(params.q_omega):
    p_col, _ = solve_price_policy(params, params.c, params.omega_grid[j])
    p_policy[:, j] = p_col
t_price = time.perf_counter() - t0
print(f"  solve_price_policy ({params.q_omega} omega states): {t_price*1000:.1f} ms  "
      f"[sequential — unaffected by thread count]\n")

d_table, c_table = precompute_demand(p_policy, params)
inv_h = (params.ns - 1) / (params.sgrid[-1] - params.sgrid[0])
s_lo  = float(params.sgrid[0])
v_by_omega = np.zeros((params.ns, params.q_omega), dtype=np.float64)
ev_all = params.p_omega @ v_by_omega.T

# ---------------------------------------------------------------------------
# Warm-up: trigger JIT compilation at max threads before timing
# ---------------------------------------------------------------------------
print("Warming up JIT (first call compiles)...")
_vfi_sweep(ev_all, d_table, c_table, p_policy, params.sgrid,
           s_lo, inv_h, params.fc, params.c, params.delta, params.beta,
           params.quad_weights)
print("Done.\n")

# ---------------------------------------------------------------------------
# Thread-scaling sweep
# ---------------------------------------------------------------------------
N_REPS = 20

thread_counts = []
n = 1
while n <= _max_threads:
    thread_counts.append(n)
    n *= 2

print(f"{'Threads':>8}  {'Per sweep (ms)':>16}  {'Speedup':>8}  "
      f"{'vs solve_price_policy':>22}")
print("-" * 62)

baseline_ms = None
for n_threads in thread_counts:
    numba.set_num_threads(n_threads)
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        _vfi_sweep(ev_all, d_table, c_table, p_policy, params.sgrid,
                   s_lo, inv_h, params.fc, params.c, params.delta, params.beta,
                   params.quad_weights)
    per_sweep_ms = (time.perf_counter() - t0) / N_REPS * 1000

    if baseline_ms is None:
        baseline_ms = per_sweep_ms
    speedup = baseline_ms / per_sweep_ms
    # How many VFI sweeps fit into the price-policy time?
    ratio = t_price * 1000 / per_sweep_ms
    print(f"{n_threads:>8}  {per_sweep_ms:>16.1f}  {speedup:>8.2f}x  "
          f"  price_policy = {ratio:>5.1f} sweeps")
