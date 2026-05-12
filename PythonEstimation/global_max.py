"""global_max.py

Tik-Tak global optimization algorithm (Guvenen et al.) for the II estimator.

Uses pre-computed grid moments from moments.csv as tik points.  The
algorithm ranks all valid grid rows by the II objective, then runs a local
solve from each of the top-K mixture points:

    x_start_k = (1 - alpha_k) * x_best + alpha_k * x_tik_k

where alpha_k decreases linearly from 1 (pure tik point) to alpha_min
(mostly current best).  x_best is updated whenever a local solve improves
the objective.

Only 10 local solves are performed (n_iterations=10) because each local
evaluation is expensive.
"""

import numpy as np
import pandas as pd

from estimation_functions import estimate_params_ii_full

# Canonical parameter order (must match estimate_params_ii_full)
_PARAM_KEYS  = ["gamma", "mu_eta", "sigma_eta2", "rho_omega",
                "sigma_nu2", "epsilon", "delta"]
# Corresponding column names in moments.csv
_GRID_COLS   = ["γ", "μη", "ση2", "ρω", "σν2", "ϵ", "δ"]
_MOMENT_COLS = ["avg_isr", "var_log1p_isr", "avg_gross_margin",
                "γ_OLS", "ρ_ω", "σ_η2", "avg_opex_sales"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_grid_objectives(df_grid, target_moments, W):
    """Compute the II objective at every valid grid row.

    Returns df_grid rows with an added ``obj_value`` column, sorted
    ascending (best first).
    """
    m_hat = np.array([target_moments[col] for col in _MOMENT_COLS])

    ok = ~df_grid["failed"].astype(bool)
    ok &= ~df_grid[_MOMENT_COLS].isna().any(axis=1)
    ok &= ~np.isinf(df_grid[_MOMENT_COLS].to_numpy(dtype=float)).any(axis=1)

    df_ok = df_grid.loc[ok].copy()
    M_all = df_ok[_MOMENT_COLS].to_numpy(dtype=np.float64)

    diff    = m_hat[None, :] - M_all
    obj_all = np.einsum("ni,ij,nj->n", diff, W, diff)

    df_ok["obj_value"] = obj_all
    return df_ok.sort_values("obj_value").reset_index(drop=True)


def _grid_row_to_theta(row):
    """Extract a structural parameter vector from a grid DataFrame row."""
    return np.array([float(row[col]) for col in _GRID_COLS])


# ---------------------------------------------------------------------------
# Tik-Tak
# ---------------------------------------------------------------------------

def tiktak(
    df_grid,
    target_moments,
    W,
    params_base,
    n_firms=500,
    n_years=20,
    seed=212311,
    max_iter=1000,
    n_iterations=10,
    alpha_min=0.1,
    verbose=True,
    method="nelder-mead",
    tol=1e-2,
    tol_final=1e-4,
):
    """Tik-Tak global optimization for the II estimator.

    Parameters
    ----------
    df_grid        : pd.DataFrame — moments.csv with pre-computed moments
    target_moments : dict — target moment values
    W              : (7, 7) weighting matrix
    params_base    : Parameters — supplies fixed fields (c, fc, beta, ns, size)
    n_firms        : int
    n_years        : int
    seed           : int
    max_iter       : int — BOBYQA function-evaluation budget per local solve
    n_iterations   : int — number of Tik-Tak iterations (default 10)
    alpha_min      : float — weight on tik point in the final iteration
                     (default 0.1; 1.0 in the first iteration)
    verbose        : bool

    Returns
    -------
    dict — same keys as ``estimate_params_ii_full``, plus
        ``tiktak_history`` : list of (iteration, obj_value, alpha) tuples
    """
    # --- Rank grid points ---
    df_sorted = _compute_grid_objectives(df_grid, target_moments, W)
    if len(df_sorted) < n_iterations:
        raise ValueError(
            f"Grid has only {len(df_sorted)} valid rows; "
            f"need at least n_iterations={n_iterations}."
        )

    if verbose:
        print("=== Tik-Tak Global Optimization ===")
        print(f"  Valid grid rows : {len(df_sorted)}")
        print(f"  Iterations      : {n_iterations}")
        print(f"  Best grid obj   : {df_sorted['obj_value'].iloc[0]:.6f}")
        print()

    # --- Initialise from best grid point ---
    x_best   = _grid_row_to_theta(df_sorted.iloc[0])
    obj_best = float(df_sorted["obj_value"].iloc[0])
    best_result = None
    history = []

    # --- Tik-Tak loop ---
    # alpha decreases linearly: 1.0 on iteration 1, alpha_min on iteration n_iterations
    for it in range(n_iterations):
        if n_iterations > 1:
            alpha = 1.0 - (1.0 - alpha_min) * it / (n_iterations - 1)
        else:
            alpha = 1.0

        x_tik   = _grid_row_to_theta(df_sorted.iloc[it])
        x_start = (1.0 - alpha) * x_best + alpha * x_tik

        if verbose:
            print(
                f"--- Iteration {it + 1}/{n_iterations}  "
                f"α={alpha:.3f}  "
                f"tik_obj={df_sorted['obj_value'].iloc[it]:.6f}  "
                f"best_so_far={obj_best:.6f} ---"
            )

        result = estimate_params_ii_full(
            target_moments,
            list(x_start),
            W,
            params_base=params_base,
            n_firms=n_firms,
            n_years=n_years,
            seed=seed,
            max_iter=max_iter,
            verbose=verbose,
            n_restarts=0,
            method=method,
            tol=tol,
        )

        history.append((it + 1, result["obj_value"], alpha))

        if result["obj_value"] < obj_best:
            obj_best    = result["obj_value"]
            x_best      = np.array([result[pname] for pname in _PARAM_KEYS])
            best_result = result
            if verbose:
                print(f"  *** New best: {obj_best:.8f} ***")
        elif verbose:
            print(f"  No improvement (obj={result['obj_value']:.8f})")

    # Guard: if no iteration improved over the grid, use the last result
    if best_result is None:
        best_result = result

    # --- Final polish at best point with tight tolerance ---
    if verbose:
        print(f"\n--- Final polish (tol={tol_final}) at best point (obj={obj_best:.8f}) ---")

    x_final = np.array([best_result[pname] for pname in _PARAM_KEYS])
    final_result = estimate_params_ii_full(
        target_moments,
        list(x_final),
        W,
        params_base=params_base,
        n_firms=n_firms,
        n_years=n_years,
        seed=seed,
        max_iter=max_iter*5,
        verbose=verbose,
        n_restarts=0,
        method=method,
        tol=tol_final,
    )
    history.append(("final", final_result["obj_value"], 0.0))

    if final_result["obj_value"] < obj_best:
        obj_best    = final_result["obj_value"]
        best_result = final_result
        if verbose:
            print(f"  *** Final polish improved: {obj_best:.8f} ***")
    elif verbose:
        print(f"  Final polish did not improve (obj={final_result['obj_value']:.8f})")

    best_result["tiktak_history"] = history

    if verbose:
        print("\n=== Tik-Tak Complete ===")
        print(f"  Best objective: {obj_best:.8f}")
        for pname in _PARAM_KEYS:
            print(f"  {pname:12s} = {best_result[pname]:.6f}")

    return best_result
