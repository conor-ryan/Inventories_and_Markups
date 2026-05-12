"""summarize_monte_carlo.py

Summarize Monte Carlo estimation outputs across all datasets.

Reads files named estimates_id_XXX.csv from --sim-data-dir and writes a
parameter-level summary CSV describing the relationship between estimates
and true values.

Usage
-----
python summarize_monte_carlo.py \
    --sim-data-dir ../../SimulatedData \
    --output-path ../../SimulatedData/monte_carlo_summary.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

PARAM_NAMES = [
    "gamma",
    "mu_eta",
    "sigma_eta2",
    "rho_omega",
    "sigma_nu2",
    "epsilon",
    "delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim-data-dir",
        default="../../SimulatedData",
        help="Directory containing estimates_id_*.csv outputs",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Where to write summary CSV. Default: "
            "<sim-data-dir>/monte_carlo_summary.csv"
        ),
    )
    return parser.parse_args()


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_var = float(np.var(x))
    y_var = float(np.var(y))
    if x_var <= 0.0 or y_var <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_slope(y: np.ndarray, x: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_center = x - x_mean
    denom = float(np.dot(x_center, x_center))
    if denom <= 0.0:
        return float("nan")
    numer = float(np.dot(x_center, y - y_mean))
    return numer / denom


def load_all_estimates(sim_data_dir: Path) -> tuple[pd.DataFrame, float, int]:
    files = sorted(sim_data_dir.glob("estimates_id_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No files matching estimates_id_*.csv found in {sim_data_dir}"
        )

    all_rows = []
    converged_values = []

    for fpath in files:
        dataset_id = fpath.stem.split("_")[-1]
        df = pd.read_csv(fpath)

        required = {"parameter", "true_value", "estimate", "std_error"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{fpath.name} missing required columns: {sorted(missing)}")

        # Grab convergence diagnostic if present.
        conv = df.loc[df["parameter"] == "converged", "estimate"]
        if not conv.empty:
            converged_values.append(float(conv.iloc[0]))

        df_param = df.loc[df["parameter"].isin(PARAM_NAMES), [
            "parameter", "true_value", "estimate", "std_error"
        ]].copy()
        df_param["dataset_id"] = dataset_id
        all_rows.append(df_param)

    panel = pd.concat(all_rows, ignore_index=True)
    convergence_rate = float(np.mean(converged_values)) if converged_values else float("nan")
    return panel, convergence_rate, len(files)


def summarize(panel: pd.DataFrame, convergence_rate: float, n_files: int) -> pd.DataFrame:
    out_rows = []

    for param in PARAM_NAMES:
        g = panel.loc[panel["parameter"] == param].copy()
        g = g.replace([np.inf, -np.inf], np.nan)
        g = g.dropna(subset=["true_value", "estimate"])

        if g.empty:
            out_rows.append(
                {
                    "parameter": param,
                    "n_obs": 0,
                    "n_datasets": n_files,
                    "convergence_rate": convergence_rate,
                    "mean_true": float("nan"),
                    "mean_estimate": float("nan"),
                    "bias": float("nan"),
                    "std_estimate": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "corr_true_estimate": float("nan"),
                    "slope_est_on_true": float("nan"),
                    "mean_std_error": float("nan"),
                    "median_std_error": float("nan"),
                    "mean_abs_bias_over_se": float("nan"),
                }
            )
            continue

        true_v = g["true_value"].to_numpy(dtype=np.float64)
        est_v = g["estimate"].to_numpy(dtype=np.float64)
        se_v = g["std_error"].to_numpy(dtype=np.float64)

        err = est_v - true_v
        abs_err = np.abs(err)
        rmse = math.sqrt(float(np.mean(err**2)))

        valid_se = np.isfinite(se_v) & (se_v > 0.0)
        if np.any(valid_se):
            mean_abs_bias_over_se = float(np.mean(np.abs(err[valid_se] / se_v[valid_se])))
            mean_se = float(np.mean(se_v[valid_se]))
            median_se = float(np.median(se_v[valid_se]))
        else:
            mean_abs_bias_over_se = float("nan")
            mean_se = float("nan")
            median_se = float("nan")

        out_rows.append(
            {
                "parameter": param,
                "n_obs": int(true_v.size),
                "n_datasets": n_files,
                "convergence_rate": convergence_rate,
                "mean_true": float(np.mean(true_v)),
                "mean_estimate": float(np.mean(est_v)),
                "bias": float(np.mean(err)),
                "std_estimate": float(np.std(est_v, ddof=1)) if est_v.size > 1 else float("nan"),
                "mae": float(np.mean(abs_err)),
                "rmse": rmse,
                "corr_true_estimate": _safe_corr(true_v, est_v),
                "slope_est_on_true": _safe_slope(est_v, true_v),
                "mean_std_error": mean_se,
                "median_std_error": median_se,
                "mean_abs_bias_over_se": mean_abs_bias_over_se,
            }
        )

    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()

    sim_data_dir = Path(args.sim_data_dir).resolve()
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else sim_data_dir / "monte_carlo_summary.csv"
    )

    panel, convergence_rate, n_files = load_all_estimates(sim_data_dir)
    summary = summarize(panel, convergence_rate, n_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"Read {n_files} estimate files from: {sim_data_dir}")
    print(f"Wrote summary CSV: {output_path}")


if __name__ == "__main__":
    main()
