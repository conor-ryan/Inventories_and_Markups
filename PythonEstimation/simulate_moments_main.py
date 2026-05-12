"""simulate_moments_main.py

Python replication of SimulateMoments.jl parameter sweep.

Defaults mirror Julia script:
- n_param_points = 2000
- n_firms = 500
- n_years = 20
- seed = 212311
- output = SimulatedData/moments.csv
"""

from pathlib import Path
import math
import time
import argparse

import numpy as np
from scipy.stats import qmc

from estimation_functions import compute_moments_on_grid


def halton_param_vectors(param_bounds, n_param_points, seed=212311):
    """Generate Halton draws mapped to parameter bounds.

    param_bounds: list of 7 (lo, hi) tuples in parameter order.
    """
    d = len(param_bounds)
    sampler = qmc.Halton(d=d, scramble=True, seed=seed)
    u = sampler.random(n=n_param_points)

    lo = np.array([b[0] for b in param_bounds], dtype=np.float64)
    hi = np.array([b[1] for b in param_bounds], dtype=np.float64)
    return lo[None, :] + u * (hi - lo)[None, :]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-param-points", type=int, default=2000)
    p.add_argument("--n-firms", type=int, default=500)
    p.add_argument("--n-years", type=int, default=20)
    p.add_argument("--seed", type=int, default=212311)
    p.add_argument("--output-path", default=None,
                   help="CSV path for moments grid; defaults to SimulatedData/moments.csv")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Worker processes (default: all logical cores)")
    p.add_argument("--threads-per-worker", type=int, default=1,
                   help="Numba threads per worker process (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()

    # Parameter bounds (same order as Julia)
    epsilon_bounds = (4.0, 20.0)
    sigma_nu2_bounds = (0.01, 0.3)
    delta_bounds = (0.005, 0.1)
    mu_eta_bounds = (math.log(0.0001), math.log(0.5))
    gamma_bounds = (0.5, 1.25)
    sigma_eta2_bounds = (0.025, 0.15)
    rho_bounds = (0.0, 0.9)

    param_bounds = [
        gamma_bounds,
        mu_eta_bounds,
        sigma_eta2_bounds,
        rho_bounds,
        sigma_nu2_bounds,
        epsilon_bounds,
        delta_bounds,
    ]

    repo_root = Path(__file__).resolve().parents[2]
    default_output = repo_root / "SimulatedData" / "moments.csv"
    output_path = Path(args.output_path) if args.output_path else default_output

    param_vectors = halton_param_vectors(
        param_bounds, args.n_param_points, seed=args.seed,
    )

    print(f"Running parameter sweep with {len(param_vectors)} points...")
    t0 = time.perf_counter()
    df_out = compute_moments_on_grid(
        param_vectors,
        n_firms=args.n_firms,
        n_years=args.n_years,
        seed=args.seed,
        output_path=output_path,
        verbose=True,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
    )
    print(f"Sweep wall time: {time.perf_counter() - t0:.1f}s")

    n_ok = int((~df_out["failed"].astype(bool)).sum())
    print(f"Sweep complete. {n_ok} / {len(df_out)} points succeeded.")
    print(f"Saved: {output_path}")

    fail_fraction = float(df_out["failed"].astype(bool).mean())
    print(f"Failure fraction: {fail_fraction:.4f} ({len(df_out) - n_ok} / {len(df_out)})")

    df_success = df_out.loc[~df_out["failed"].astype(bool)]
    if len(df_success) == 0:
        print("No successful simulations; moment summaries unavailable.")
        return

    print("\nMoment summaries (successful simulations only):")
    print("moment, p25, median, mean, p75")
    moment_cols = [
        "avg_isr", "var_log1p_isr", "avg_gross_margin",
        "γ_OLS", "ρ_ω", "σ_η2", "avg_opex_sales",
    ]
    for col in moment_cols:
        vals = df_success[col].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            print(f"{col}, NaN, NaN, NaN, NaN")
        else:
            q25, q50, q75 = np.quantile(vals, [0.25, 0.50, 0.75])
            print(f"{col}, {q25:.6f}, {q50:.6f}, {vals.mean():.6f}, {q75:.6f}")


if __name__ == "__main__":
    main()
