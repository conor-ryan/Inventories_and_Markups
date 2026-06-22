"""simulatemoments.py

Python translation of SimulateMoments.jl.

Generates a Halton parameter grid, slices it by job-array task ID, and
evaluates all 7 estimation moments sequentially over the slice.  Inner
parallelism (Numba threads) handles each solve_value_function call.
Outer parallelism is via the job array.

Supports PBS Pro and SLURM job arrays.
  PBS  : reads PBS_ARRAY_INDEX (task id) and PBS_ARRAY_N_TASKS (total tasks,
         set explicitly in the PBS script via -v PBS_ARRAY_N_TASKS=N).
  SLURM: reads SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT.
  None : runs as a single task over all parameter vectors.

Usage (single node):
    python simulatemoments.py

Usage (PBS Pro job array):
    qsub -J 1-N -v PBS_ARRAY_N_TASKS=N simulatemoments.pbs

Usage (SLURM job array):
    #SBATCH --array=1-N
    python simulatemoments.py
"""

import os
import math

import numpy as np

from simulation_functions import halton_param_vectors, compute_moments_on_grid

# ---------------------------------------------------------------------------
# Parameter bounds  (must match SimulateMoments.jl)
# ---------------------------------------------------------------------------
# Order: [γ, μω→μη, ση2, ρ, σν2, ε, δ]
# μω is drawn then converted: μη = log(μω) * (1 − ρ)
BOUNDS = [
    (4.0,  30.0),   # ε
    (0.01,  0.4),   # σν2
    (0.005, 0.2),   # δ
    (0.01,  0.2),   # μω  (converted to μη below)
    (0.8,   1.5),   # γ
    (0.1,   1.0),   # ση2
    (0.00,  0.9),   # ρ
]

# Matching the field order expected by compute_moments_on_grid:
# [γ, μη, ση2, ρω, σν2, ε, δ]
# We draw in BOUNDS order then rearrange after the μω→μη transform.

N_PARAM_POINTS = 2500
SEED           = 212311

N_FIRMS   = 500
N_YEARS   = 20
BURN_IN   = 100
GRID_SIZE = 200
SCALE     = 1.0
SIZE      = 100.0
SOLVE_TOL = 1e-2
MAX_VFI   = 500

OUT_DIR ="SimulatedData"

# ---------------------------------------------------------------------------
# Generate Halton draws in BOUNDS order
# ---------------------------------------------------------------------------
raw_vectors = halton_param_vectors(BOUNDS, N_PARAM_POINTS, seed=SEED)

# BOUNDS order: [ε, σν2, δ, μω, γ, ση2, ρ]
# Apply μω → μη = log(μω) * (1 − ρ)  (index 3 = μω, index 6 = ρ)
param_vectors = []
for v in raw_vectors:
    eps_i   = v[0]
    snu2_i  = v[1]
    delta_i = v[2]
    mu_omega_i = v[3]
    gamma_i = v[4]
    sig2_i  = v[5]
    rho_i   = v[6]
    mu_eta_i = math.log(mu_omega_i) * (1.0 - rho_i)
    # Target order for compute_moments_on_grid: [γ, μη, ση2, ρω, σν2, ε, δ]
    param_vectors.append(np.array([gamma_i, mu_eta_i, sig2_i, rho_i, snu2_i, eps_i, delta_i]))

# ---------------------------------------------------------------------------
# Job-array slicing  (PBS Pro and SLURM)
# ---------------------------------------------------------------------------
if "PBS_ARRAY_INDEX" in os.environ:
    task_id = int(os.environ["PBS_ARRAY_INDEX"])
    n_tasks = int(os.environ.get("PBS_ARRAY_N_TASKS", "1"))
elif "SLURM_ARRAY_TASK_ID" in os.environ:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
else:
    task_id = 1
    n_tasks = 1

chunk_size = math.ceil(len(param_vectors) / n_tasks)
i_start    = (task_id - 1) * chunk_size          # 0-based
i_end      = min(task_id * chunk_size, len(param_vectors))
param_slice = param_vectors[i_start:i_end]

if n_tasks > 1:
    output_path = os.path.join(OUT_DIR, f"moments_{task_id:03d}.csv")
else:
    output_path = os.path.join(OUT_DIR, "moments.csv")

print(
    f"Task {task_id}/{n_tasks}: grid points {i_start+1}–{i_end} "
    f"({len(param_slice)} total)"
)

# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------
df_out = compute_moments_on_grid(
    param_slice,
    n_firms=N_FIRMS,
    n_years=N_YEARS,
    seed=SEED,
    output_path=output_path,
    max_value_iterations=MAX_VFI,
    grid_size=GRID_SIZE,
    scale=SCALE,
    size=SIZE,
    solve_tol=SOLVE_TOL,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_ok = int((~df_out["failed"]).sum())
print(f"Sweep complete. {n_ok} / {len(df_out)} points succeeded.")
print(f"Saved: {output_path}")

fail_frac = df_out["failed"].mean()
print(f"Failure fraction: {fail_frac:.4f} ({df_out['failed'].sum()} / {len(df_out)})")

df_success = df_out[(~df_out["failed"]) & (df_out["avg_isr"] < 4.0)]
if len(df_success) == 0:
    print("No successful simulations; moment summaries unavailable.")
else:
    print(f"\nMoment summaries (successful simulations, n={len(df_success)}):")
    print(f"{'moment':<22} {'min':>10} {'p10':>10} {'p25':>10} {'median':>10} {'p75':>10} {'p90':>10} {'max':>10}")
    moment_cols = [
        "avg_isr", "var_log1p_isr", "avg_gross_margin",
        "gamma_OLS", "rho_omega_ar1", "sigma_eta2_ar1", "avg_opex_sales",
    ]
    for col in moment_cols:
        vals = df_success[col].dropna()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            print(f"  {col:<20}  all NaN")
            continue
        qs = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
        print(
            f"  {col:<20}  "
            f"{vals.min():10.6f} {qs[0]:10.6f} {qs[1]:10.6f} "
            f"{qs[2]:10.6f} {qs[3]:10.6f} {qs[4]:10.6f} {vals.max():10.6f}"
        )
