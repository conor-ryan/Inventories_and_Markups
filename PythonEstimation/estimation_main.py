"""estimation_main.py

Stage 3 Monte Carlo: estimate the model for a single simulated dataset.

Usage
-----
    python estimation_main.py --id 003 --n-threads 8 --sim-data-dir ../../SimulatedData

Arguments
---------
--id           : zero-padded 3-digit dataset identifier, e.g. 001
--n-threads    : number of Numba/OpenMP threads for the VFI inner loop
--sim-data-dir : path to the SimulatedData directory (relative or absolute)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parse args first so --n-threads reaches numba before any JIT compilation
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--id",            required=True,
                   help="Dataset ID, e.g. 003")
    p.add_argument("--n-threads",     type=int, default=1,
                   help="Numba thread count for VFI parallelism")
    p.add_argument("--sim-data-dir",  default="../../SimulatedData",
                   help="Path to SimulatedData directory")
    return p.parse_args()


args = _parse_args()

import numba
numba.set_num_threads(args.n_threads)

from model_functions import Parameters, solve_value_function
from estimation_functions import (
    simulate_all_moments,
    estimate_params_ii_full,
    compute_ii_asymptotic_variance,
)
from global_max import tiktak


# ---------------------------------------------------------------------------
# Fixed structural parameters (SolveModel.jl line 6 defaults)
# Estimated parameters (gamma, mu_eta, ...) are overridden inside the
# objective — only the structural/fixed fields matter here.
# ---------------------------------------------------------------------------

PARAMS_BASE = Parameters(
    c=1.0,
    fc=0.0,
    beta=0.95,
    mu_nu=1.0,
    sigma_nu2=0.15,   # placeholder — overridden in objective
    ns=200,
    scale=1.0,
    size=100.0,
    mu_eta=math.log(0.01),  # placeholder — overridden in objective
    sigma_eta2=0.05,
    rho_omega=0.1,
    gamma=0.9,
    delta=0.01,
    epsilon=8.0,
)

MOMENT_NAMES = ["avg_isr", "var_log1p_isr", "avg_gross_margin",
                "γ_OLS", "ρ_ω", "σ_η2", "avg_opex_sales"]
PARAM_NAMES  = ["gamma", "mu_eta", "sigma_eta2", "rho_omega",
                "sigma_nu2", "epsilon", "delta"]

# Number of firms used to compute target moments in the simulated datasets
# (must match the Julia simulation that generated the data)
SAMPLE_SIZE = 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_id  = args.id.zfill(3)
    sim_data    = Path(args.sim_data_dir).resolve()
    output_path = sim_data / f"estimates_id_{dataset_id}.csv"

    print(f"=== Monte Carlo estimation: dataset {dataset_id} ===")
    print(f"  sim_data   : {sim_data}")
    print(f"  n_threads  : {args.n_threads}")
    print(f"  output     : {output_path}")

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    df_target = pd.read_csv(sim_data / f"target_moments_id_{dataset_id}.csv")
    target_moments = dict(zip(df_target["moment"], df_target["value"].astype(float)))

    df_vcov = pd.read_csv(sim_data / f"target_moment_vcov_id_{dataset_id}.csv",
                          index_col=0)
    vcov = df_vcov.loc[MOMENT_NAMES, MOMENT_NAMES].to_numpy(dtype=np.float64)
    W    = np.linalg.inv(vcov)

    df_true    = pd.read_csv(sim_data / f"true_parameters_id_{dataset_id}.csv")
    true_vals  = {
        "gamma":      float(df_true["γ"].iloc[0]),
        "mu_eta":     float(df_true["μη"].iloc[0]),
        "sigma_eta2": float(df_true["ση2"].iloc[0]),
        "rho_omega":  float(df_true["ρ_ω"].iloc[0]),
        "sigma_nu2":  float(df_true["σν2"].iloc[0]),
        "epsilon":    float(df_true["ϵ"].iloc[0]),
        "delta":      float(df_true["δ"].iloc[0]),
    }

    df_grid = pd.read_csv(sim_data / "moments.csv")

    print("\nTarget moments:")
    for k in MOMENT_NAMES:
        print(f"  {k:22s} = {target_moments[k]:.6f}")

    # ------------------------------------------------------------------
    # 3. Estimate
    # ------------------------------------------------------------------
    print("\nRunning Tik-Tak estimation ...")
    import time
    t0 = time.perf_counter()
    ii_result = tiktak(
        df_grid,
        target_moments,
        W,
        params_base=PARAMS_BASE,
        n_firms=500,
        n_years=20,
        seed=212311,
        max_iter=5000,
        n_iterations=10,
        alpha_min=0.1,
        verbose=False,
        method="bobyqa",
        tol=1e-2,
        tol_final=1e-5,
    )
    t_est = time.perf_counter() - t0
    print(f"  Estimation wall time : {t_est:.1f}s")
    print(f"  Objective            : {ii_result['obj_value']:.8f}")
    print("  Tik-Tak history:")
    for rec in ii_result["tiktak_history"]:
        print(f"    iter {str(rec[0]):>5s}  obj={rec[1]:.8f}  alpha={rec[2]:.3f}")

    # ------------------------------------------------------------------
    # 4. Standard errors
    # ------------------------------------------------------------------
    params_hat = Parameters(
        c=PARAMS_BASE.c,     fc=PARAMS_BASE.fc,
        beta=PARAMS_BASE.beta, mu_nu=PARAMS_BASE.mu_nu,
        ns=PARAMS_BASE.ns,   size=PARAMS_BASE.size,
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
        sample_size=SAMPLE_SIZE,
        verbose=False,
    )
    print(f"  SE wall time         : {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 5. Build and save results
    # ------------------------------------------------------------------
    rows = []
    for name, se in zip(PARAM_NAMES, se_results["se"]):
        rows.append({
            "parameter":  name,
            "true_value": true_vals[name],
            "estimate":   ii_result[name],
            "std_error":  se,
        })
    # Scalar diagnostics as extra rows
    rows.append({"parameter": "obj_value",  "true_value": float("nan"),
                 "estimate": ii_result["obj_value"], "std_error": float("nan")})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    # Summary table
    print(f"\n{'parameter':14s}  {'true':>12s}  {'estimate':>12s}  "
          f"{'std_error':>12s}  {'bias/SE':>10s}")
    print("-" * 68)
    for name, se in zip(PARAM_NAMES, se_results["se"]):
        est     = ii_result[name]
        true    = true_vals[name]
        bias_se = (est - true) / se if se > 0 else float("nan")
        print(f"  {name:12s}  {true:12.6f}  {est:12.6f}  {se:12.6f}  {bias_se:10.3f}")


if __name__ == "__main__":
    main()
