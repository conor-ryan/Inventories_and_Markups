from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from PythonEstimation.estimation_functions import (
    MOMENT_NAMES,
    compute_full_ii_asymptotic_variance,
    estimate_params_ii_full,
    select_best_grid_start,
)
from PythonEstimation.model_functions import Parameters


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full II estimation from saved moments files.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing input/output CSV files. Defaults to <project_root>/SimulatedData.")
    parser.add_argument("--target-moments-file", type=str, default="target_moments.csv", help="Target moments CSV filename.")
    parser.add_argument("--target-vcov-file", type=str, default="target_moment_vcov.csv", help="Target moment covariance CSV filename.")
    parser.add_argument("--grid-file", type=str, default="moments.csv", help="Precomputed grid moments CSV filename.")
    parser.add_argument("--results-file", type=str, default="estimated_parameters.csv", help="Output CSV filename for parameter estimates and standard errors.")
    parser.add_argument("--seed", type=int, default=212311, help="Random seed used during estimation and standard-error simulation.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size used to scale asymptotic variance for standard errors.")
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    out_dir = args.data_dir if args.data_dir is not None else (project_root / "SimulatedData")

    target_moments_path = out_dir / args.target_moments_file
    target_vcov_path = out_dir / args.target_vcov_file
    grid_path = out_dir / args.grid_file
    results_path = out_dir / args.results_file

    seed = args.seed
    sample_size = args.sample_size

    print(f"Using data directory: {out_dir}")
    print(f"Target moments file: {target_moments_path}")
    print(f"Target vcov file: {target_vcov_path}")
    print(f"Grid file: {grid_path}")
    print(f"Results file: {results_path}")

    print("Loading saved target moments, moment covariance matrix, and grid...")
    df_target_moments = pd.read_csv(target_moments_path)
    df_target_vcov = pd.read_csv(target_vcov_path)
    df_grid = pd.read_csv(grid_path)

    moment_labels = list(MOMENT_NAMES)
    if not (df_target_moments["moment"].tolist() == moment_labels and df_target_vcov["moment"].tolist() == moment_labels):
        raise RuntimeError("target moments file has unexpected moment ordering")

    target_moments = {name: float(value) for name, value in zip(moment_labels, df_target_moments["value"].to_numpy(dtype=float))}

    vcov = np.zeros((len(moment_labels), len(moment_labels)), dtype=float)
    for j, name in enumerate(moment_labels):
        vcov[:, j] = df_target_vcov[name].to_numpy(dtype=float)
    w = np.linalg.inv(vcov)

    print("Selecting initial guess from precomputed grid...")
    start_guess = select_best_grid_start(df_grid, target_moments, w)
    print(f"Best pre-computed grid objective value: {start_guess['obj_value']:.6f}")

    best_row = int(start_guess["row_index"])
    print("Moments at best pre-computed grid point:")
    for name in moment_labels:
        print(f"  {name:16s} = {float(df_grid.loc[best_row, name]):10.6f}")

    print("Estimating parameters...")
    ii_full = estimate_params_ii_full(
        target_moments,
        np.array(
            [
                start_guess["γ"],
                start_guess["μη"],
                start_guess["ση2"],
                start_guess["ρω"],
                start_guess["σν2"],
                start_guess["ϵ"],
                start_guess["δ"],
            ],
            dtype=float,
        ),
        w,
        n_firms=5000,
        n_years=20,
        max_iter=500,
        seed=seed,
        verbose=True,
        g_abstol=1e-1,
    )

    params_hat = Parameters(
        mu_eta=ii_full["μη"],
        sigma_eta2=ii_full["ση2"],
        rho_omega=ii_full["ρω"],
        gamma=ii_full["γ"],
        delta=ii_full["δ"],
        epsilon=ii_full["ϵ"],
        sigma_nu2=ii_full["σν2"],
    )

    print("Computing standard errors...")
    se_results = compute_full_ii_asymptotic_variance(
        params_hat,
        w,
        n_firms=5000,
        n_years=20,
        seed=seed,
        solve_maxiter=1000,
        sample_size=sample_size,
    )

    df_results = pd.DataFrame(
        {
            "parameter": ["γ", "μη", "ση2", "ρω", "σν2", "ϵ", "δ"],
            "estimate": [ii_full["γ"], ii_full["μη"], ii_full["ση2"], ii_full["ρω"], ii_full["σν2"], ii_full["ϵ"], ii_full["δ"]],
            "std_error": se_results["se"],
        }
    )

    df_results.to_csv(results_path, index=False)
    print(f"Saved estimates to {results_path}")


if __name__ == "__main__":
    run()
