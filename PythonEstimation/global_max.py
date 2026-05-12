"""global_max.py

Two surrogate warm-start methods for indirect inference estimation, using a
pre-computed grid of (parameter, simulated-moment) pairs.

Methods
-------
softmin_warm_start
    Boltzmann-weighted centroid of the n_top best grid points.
    Stays strictly inside the convex hull of the selected points so it can
    never land on a global parameter bound.  Recommended default.

quadratic_warm_start
    Fits a full quadratic surrogate in *normalised* parameter space to the
    n_top best grid points and returns the analytic minimum, clipped to the
    bounding box of those points (the "trust region").
    Requires n_top ≫ 36 (number of quadratic features for 7 parameters).

Usage
-----
    from global_max import softmin_warm_start, quadratic_warm_start
    guess, info = softmin_warm_start(df_grid, target_moments, W)
"""

import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column names matching SimulateMoments.jl / compute_moments_on_grid output
# ---------------------------------------------------------------------------
PARAM_COLS  = ["γ", "μη", "ση2", "ρω", "σν2", "ϵ", "δ"]
PARAM_NAMES = ["gamma", "mu_eta", "sigma_eta2", "rho_omega",
               "sigma_nu2", "epsilon", "delta"]
MOMENT_COLS = ["avg_isr", "var_log1p_isr", "avg_gross_margin",
               "γ_OLS", "ρ_ω", "σ_η2", "avg_opex_sales"]

_BOUNDS = {
    "gamma":      (0.05,   3.0),
    "mu_eta":     (-5.0,   5.0),
    "sigma_eta2": (1e-6,   5.0),
    "rho_omega":  (-0.999, 0.999),
    "sigma_nu2":  (1e-6,   5.0),
    "epsilon":    (1.1,    20.0),
    "delta":      (0.001,  0.999),
}

_N_PARAMS    = len(PARAM_COLS)                                          # 7
_N_QUAD_FEAT = 1 + _N_PARAMS + _N_PARAMS * (_N_PARAMS + 1) // 2       # 36


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_objectives(df_grid, target_moments, W):
    """Vectorised weighted-II objective for all non-failed grid rows.

    Returns
    -------
    ok_rows : pd.DataFrame — non-failed rows with reset integer index
    obj     : (n_ok,) array — objective for each row of ok_rows
    """
    m_hat   = np.array([target_moments[k] for k in MOMENT_COLS])
    ok      = ~df_grid["failed"].astype(bool)
    ok      = ok & ~df_grid[MOMENT_COLS].isna().any(axis=1)
    ok      = ok & ~np.isinf(df_grid[MOMENT_COLS].to_numpy(dtype=np.float64)).any(axis=1)
    ok_rows = df_grid[ok].reset_index(drop=True)
    M       = ok_rows[MOMENT_COLS].to_numpy(dtype=np.float64)
    diff    = m_hat[None, :] - M
    obj     = np.einsum("ni,ij,nj->n", diff, W, diff)
    return ok_rows, obj


def _select_top(ok_rows, obj, n_top):
    """Return (theta_mat, y) for the n_top lowest-objective rows."""
    n_use     = min(n_top, len(obj))
    order     = np.argsort(obj)[:n_use]
    theta_mat = ok_rows.iloc[order][PARAM_COLS].to_numpy(dtype=np.float64)
    y         = obj[order]
    return theta_mat, y


def _build_quad_features(theta_norm):
    """Build full quadratic design matrix with intercept.

    For n_params = 7:  1 + 7 + 28 = 36 columns.

    Parameters
    ----------
    theta_norm : (n, 7) *normalised* parameter matrix

    Returns
    -------
    X        : (n, 36) design matrix
    quad_idx : list of (i, j) pairs, one per quadratic column
    """
    n, p = theta_norm.shape
    cols = [np.ones((n, 1)), theta_norm]
    quad_idx = []
    for i in range(p):
        for j in range(i, p):
            cols.append((theta_norm[:, i] * theta_norm[:, j])[:, None])
            quad_idx.append((i, j))
    return np.hstack(cols), quad_idx


# ---------------------------------------------------------------------------
# Public API — Softmin centroid
# ---------------------------------------------------------------------------

def softmin_warm_start(df_grid, target_moments, W, n_top=500, verbose=True):
    """Boltzmann-weighted centroid of the n_top lowest-objective grid points.

    Algorithm
    ---------
    1. Compute the weighted-II objective for all non-failed grid rows.
    2. Select the n_top rows with the lowest objective.
    3. Compute Boltzmann weights w_i ∝ exp(−λ (obj_i − obj_min)), where λ is
       chosen adaptively so the best point receives e⁴ ≈ 55× more weight than
       the median of the n_top points.
    4. Return the weighted centroid θ* = Σ w_i θ_i.

    θ* is always inside the convex hull of the selected points, so it can
    never land on a global parameter bound.

    Parameters
    ----------
    df_grid        : pd.DataFrame with PARAM_COLS + MOMENT_COLS + "failed"
    target_moments : dict keyed by MOMENT_COLS
    W              : (7, 7) weighting matrix
    n_top          : int — number of best grid points to include
    verbose        : bool

    Returns
    -------
    init_guess : length-7 list [gamma, mu_eta, sigma_eta2, rho_omega,
                 sigma_nu2, epsilon, delta]
    info : dict — theta_star, theta_grid_best, obj_grid_best,
                  lambda_, effective_n, n_fit
    """
    ok_rows, obj         = _compute_objectives(df_grid, target_moments, W)
    theta_mat, y         = _select_top(ok_rows, obj, n_top)
    n_use                = len(y)
    obj_grid_best        = float(y[0])
    theta_grid_best      = theta_mat[0].copy()

    obj_min    = float(y[0])
    obj_median = float(np.median(y))
    spread     = obj_median - obj_min

    if spread > 1e-12:
        lam = 4.0 / spread   # top-1 gets e^4 ≈ 55× weight of median
    else:
        lam = 0.0            # degenerate: uniform centroid

    log_w  = -lam * (y - obj_min)
    log_w -= log_w.max()     # numerical stability
    w      = np.exp(log_w)
    w     /= w.sum()

    theta_star  = (w[:, None] * theta_mat).sum(axis=0)
    effective_n = float(1.0 / np.sum(w ** 2))

    if verbose:
        print(f"Softmin centroid: n_top={n_use},  eff_n={effective_n:.1f},  "
              f"λ={lam:.4g}")
        print(f"  objective at grid best = {obj_grid_best:.6f}")

    info = {
        "theta_star":      theta_star,
        "theta_grid_best": theta_grid_best,
        "obj_grid_best":   obj_grid_best,
        "lambda_":         lam,
        "effective_n":     effective_n,
        "n_fit":           n_use,
    }
    return list(theta_star), info


# ---------------------------------------------------------------------------
# Public API — Trust-region quadratic
# ---------------------------------------------------------------------------

def quadratic_warm_start(df_grid, target_moments, W, n_top=500, verbose=True):
    """Quadratic surrogate warm-start with normalisation and trust-region clipping.

    Why the naive version goes to the bounds
    -----------------------------------------
    The 7 parameters have very different magnitudes (delta ≈ 0.01, epsilon ≈ 8,
    mu_eta ≈ −4.6 …).  Without normalisation the quadratic OLS is poorly
    conditioned and the fitted minimum can be far outside the data cloud,
    causing clipping to the global bounds.

    Fixes applied here
    ------------------
    1. *Normalise*: z-score each parameter dimension within the selected top-k
       set before OLS, giving equal weight to all directions.
    2. *Trust-region clip*: clip θ* to the axis-aligned bounding box of the
       selected top-k points, not to the global _BOUNDS.  The quadratic is
       only meaningful inside the region where the grid data are dense.
    3. *Fallback*: if the Hessian is not PD (non-convex surrogate), the nearest
       PD matrix (eigenvalue clamping) is used.

    Minimum recommended n_top: ~180 (≥5× the 36 quadratic features for 7
    parameters).  With n_top = 50 there are only 14 residual df.

    Parameters
    ----------
    df_grid        : pd.DataFrame with PARAM_COLS + MOMENT_COLS + "failed"
    target_moments : dict keyed by MOMENT_COLS
    W              : (7, 7) weighting matrix
    n_top          : int
    verbose        : bool

    Returns
    -------
    init_guess : length-7 list
    info : dict — theta_star, theta_grid_best, obj_grid_best,
                  obj_surrogate, r2, n_fit, hessian_pd, trust_region_active
    """
    ok_rows, obj    = _compute_objectives(df_grid, target_moments, W)
    theta_mat, y    = _select_top(ok_rows, obj, n_top)
    n_use           = len(y)

    if n_use < _N_QUAD_FEAT + 1:
        warnings.warn(
            f"quadratic_warm_start: only {n_use} valid points, but the "
            f"quadratic has {_N_QUAD_FEAT} features.  Increase n_top or use "
            "softmin_warm_start instead.",
            stacklevel=2,
        )

    obj_grid_best   = float(y[0])
    theta_grid_best = theta_mat[0].copy()

    # Trust region: bounding box of the n_top selected points
    tr_lo = theta_mat.min(axis=0)
    tr_hi = theta_mat.max(axis=0)

    # Normalise to zero mean, unit SD within the top-k set.
    # Essential when parameters differ by orders of magnitude.
    mu_fit   = theta_mat.mean(axis=0)
    std_fit  = theta_mat.std(axis=0)
    std_fit  = np.where(std_fit < 1e-10, 1.0, std_fit)
    theta_norm = (theta_mat - mu_fit) / std_fit

    # Quadratic OLS in normalised space
    X, quad_idx = _build_quad_features(theta_norm)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat  = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Extract Hessian H and linear gradient α in normalised space
    # f(θ̃) = c + α'θ̃ + Σ_{i≤j} β_{ij} θ̃_i θ̃_j
    alpha_lin = coeffs[1 : 1 + _N_PARAMS]
    H = np.zeros((_N_PARAMS, _N_PARAMS))
    for k, (i, j) in enumerate(quad_idx):
        b = coeffs[1 + _N_PARAMS + k]
        if i == j:
            H[i, i] = 2.0 * b
        else:
            H[i, j] = b
            H[j, i] = b

    evals      = np.linalg.eigvalsh(H)
    hessian_pd = bool(np.all(evals > 0))

    if hessian_pd:
        theta_norm_star = np.linalg.solve(H, -alpha_lin)
    else:
        evals_c, evecs = np.linalg.eigh(H)
        H_pd = evecs @ np.diag(np.maximum(evals_c, 1e-8)) @ evecs.T
        try:
            theta_norm_star = np.linalg.solve(H_pd, -alpha_lin)
        except np.linalg.LinAlgError:
            theta_norm_star = np.zeros(_N_PARAMS)

    # Back-transform to original space
    theta_star = theta_norm_star * std_fit + mu_fit

    # Clip to trust region (top-k bounding box), then global bounds as backstop
    trust_region_active = bool(
        np.any(theta_star < tr_lo - 1e-10) or np.any(theta_star > tr_hi + 1e-10)
    )
    theta_star = np.clip(theta_star, tr_lo, tr_hi)
    for k, name in enumerate(PARAM_NAMES):
        lo, hi = _BOUNDS[name]
        theta_star[k] = float(np.clip(theta_star[k], lo, hi))

    # Evaluate surrogate at the (now clipped) θ*
    theta_norm_eval = (theta_star - mu_fit) / std_fit
    X_star = np.concatenate(
        [[1.0], theta_norm_eval,
         [theta_norm_eval[qi] * theta_norm_eval[qj] for qi, qj in quad_idx]]
    )
    obj_surrogate = float(coeffs @ X_star)

    if verbose:
        tr_str = " [trust-region boundary active]" if trust_region_active else ""
        print(f"Quadratic surrogate: R²={r2:.4f},  n_fit={n_use},  "
              f"Hessian PD={hessian_pd}{tr_str}")
        print(f"  objective at grid best = {obj_grid_best:.6f}")
        print(f"  surrogate at θ*        = {obj_surrogate:.6f}")
    elif trust_region_active:
        warnings.warn(
            "quadratic_warm_start: analytic minimum is outside the top-k "
            "bounding box — clipped to trust region.  "
            "Consider softmin_warm_start instead.",
            stacklevel=2,
        )

    info = {
        "theta_star":          theta_star,
        "theta_grid_best":     theta_grid_best,
        "obj_grid_best":       obj_grid_best,
        "obj_surrogate":       obj_surrogate,
        "r2":                  r2,
        "n_fit":               n_use,
        "hessian_pd":          hessian_pd,
        "trust_region_active": trust_region_active,
    }
    return list(theta_star), info


# ---------------------------------------------------------------------------
# Public API — Focused Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def focused_lhs_search(df_grid, target_moments, W, n_top=50, n_sample=20,
                       tail_pct=0.10, seed=None):
    """Latin Hypercube sample within the parameter neighbourhood of the grid optimum.

    Algorithm
    ---------
    1. Compute the weighted-II objective for all non-failed grid rows.
    2. Select the n_top rows with the lowest objective.
    3. For each parameter dimension compute the tail_pct–(1−tail_pct) quantile
       range across those n_top points.  This gives a tight “promising box”
       that discards outlier corners of the top-k bounding box.
    4. Draw n_sample points via Latin Hypercube Sampling inside that box.
       LHS divides each dimension into n_sample equal-width slices and draws
       exactly one point from each slice, with slices randomly matched across
       dimensions.  This guarantees even marginal coverage with no clustering
       — far better than pure random for small n_sample.

    No model solves are performed here; the caller evaluates each proposal.

    Parameters
    ----------
    df_grid        : pd.DataFrame
    target_moments : dict
    W              : (7, 7) weighting matrix
    n_top          : int   — number of best grid points used to define the box
    n_sample       : int   — number of LHS proposals to generate
    tail_pct       : float — quantile fraction to trim from each edge of the
                     top-k bounding box (0.10 = use 10th–90th percentile)
    seed           : int or None

    Returns
    -------
    proposals : list of n_sample length-7 lists
                [gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta]
    info : dict — lo, hi (per-parameter bounds used), obj_grid_best, n_top_used
    """
    ok_rows, obj  = _compute_objectives(df_grid, target_moments, W)
    theta_mat, y  = _select_top(ok_rows, obj, n_top)
    n_use         = len(y)
    obj_grid_best = float(y[0])

    # Quantile-trimmed bounding box
    q_lo = float(np.clip(tail_pct, 0.0, 0.49))
    q_hi = 1.0 - q_lo
    if n_use >= 4:
        lo = np.quantile(theta_mat, q_lo, axis=0)
        hi = np.quantile(theta_mat, q_hi, axis=0)
    else:
        lo = theta_mat.min(axis=0)
        hi = theta_mat.max(axis=0)

    # Ensure non-degenerate intervals; fall back to 0.5% of global range
    for k, name in enumerate(PARAM_NAMES):
        g_lo, g_hi = _BOUNDS[name]
        if hi[k] <= lo[k] + 1e-10:
            mid   = 0.5 * (lo[k] + hi[k])
            gap   = 0.005 * (g_hi - g_lo)
            lo[k] = max(g_lo, mid - gap)
            hi[k] = min(g_hi, mid + gap)

    # Latin Hypercube Sample
    rng     = np.random.default_rng(seed)
    samples = np.zeros((n_sample, _N_PARAMS))
    for j in range(_N_PARAMS):
        perm           = rng.permutation(n_sample)
        u              = (perm + rng.random(n_sample)) / n_sample  # uniform (0,1)
        samples[:, j]  = lo[j] + u * (hi[j] - lo[j])

    # Backstop: clip to global bounds
    for k, name in enumerate(PARAM_NAMES):
        g_lo, g_hi    = _BOUNDS[name]
        samples[:, k] = np.clip(samples[:, k], g_lo, g_hi)

    proposals = [list(samples[i]) for i in range(n_sample)]
    info = {
        "lo":            lo,
        "hi":            hi,
        "obj_grid_best": obj_grid_best,
        "n_top_used":    n_use,
    }
    return proposals, info
