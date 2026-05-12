"""estimation_functions.py

Python replication of EstimationFunctions.jl and simulate_firm from
ModelFunctions.jl.  All functions follow the Stage 2 implementation plan
in README.md.

Stage 1 (solve_value_function) lives in model_functions.py.
"""

import math
import numpy as np
from numba import jit, prange

from model_functions import Parameters, solve_value_function


# ---------------------------------------------------------------------------
# Compiled uniform-grid policy interpolation helper
# ---------------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def _interp_row(mat, row_idx, x, x_lo, inv_h, n):
    """Linear interpolation (+ linear extrapolation) of mat[row_idx, :] at x.

    mat      : (n_omega, ns) contiguous float64 array — policy transposed
    row_idx  : integer omega state index
    x        : query point (inventory level)
    x_lo     : sgrid[0]
    inv_h    : (ns - 1) / (sgrid[-1] - sgrid[0])
    n        : ns (number of grid points)
    """
    t  = (x - x_lo) * inv_h
    i0 = int(t)
    if t <= 0.0:
        return mat[row_idx, 0] + t * (mat[row_idx, 1] - mat[row_idx, 0])
    elif i0 >= n - 1:
        excess = t - (n - 1)
        return mat[row_idx, n - 1] + excess * (mat[row_idx, n - 1] - mat[row_idx, n - 2])
    else:
        alpha = t - i0
        return mat[row_idx, i0] + alpha * (mat[row_idx, i0 + 1] - mat[row_idx, i0])


# ---------------------------------------------------------------------------
# Compiled simulation kernel
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _simulate_firms(
    p_policy_T,       # (n_omega, ns)  — p_policy transposed, contiguous
    n_policy_T,       # (n_omega, ns)  — n_policy transposed, contiguous
    sgrid,            # (ns,)
    s_lo,             # float  — sgrid[0]
    inv_h,            # float  — (ns-1) / (sgrid[-1] - sgrid[0])
    omega_grid,       # (n_omega,)
    p_omega_cumsum,   # (n_omega, n_omega) — row-wise cumsum of transition matrix
    pi_omega_cumsum,  # (n_omega,)         — cumsum of ergodic distribution
    nu_draws,         # (n_firms, burn_in + n_months) — pre-drawn lognormal demand shocks
    omega_u_draws,    # (n_firms, burn_in + n_months) — pre-drawn U[0,1] for omega transitions
    omega_init_u,     # (n_firms,)  — U[0,1] for initial omega draw
    s_init_idx,       # (n_firms,)  int64 — initial inventory grid index
    n_firms, n_months, burn_in,
    epsilon, delta, gamma,
):
    """Simulate n_firms firms for burn_in + n_months periods.

    Returns four arrays each of shape (n_firms, n_months):
        inv_out — beginning-of-period inventory
        dem_out — realized demand
        rev_out — revenue (p * D)
        exp_out — operating expense (omega * D^gamma)
    """
    ns      = len(sgrid)
    n_omega = len(omega_grid)
    neg_eps = -epsilon

    inv_out = np.empty((n_firms, n_months))
    dem_out = np.empty((n_firms, n_months))
    rev_out = np.empty((n_firms, n_months))
    exp_out = np.empty((n_firms, n_months))

    for firm in prange(n_firms):

        # ---- initial state ----
        s = sgrid[s_init_idx[firm]]

        # Initial omega: inverse CDF on ergodic distribution
        u       = omega_init_u[firm]
        omega_j = n_omega - 1
        for k in range(n_omega):
            if u <= pi_omega_cumsum[k]:
                omega_j = k
                break

        # ---- burn-in (discard output) ----
        for t in range(burn_in):
            p  = _interp_row(p_policy_T, omega_j, s, s_lo, inv_h, ns)
            n  = _interp_row(n_policy_T, omega_j, s, s_lo, inv_h, ns)
            nu = nu_draws[firm, t]
            D  = min(nu * p ** neg_eps, s)
            s  = max((1.0 - delta) * (s - D + n), 0.0)

            # Markov transition for omega
            u     = omega_u_draws[firm, t]
            new_j = n_omega - 1
            for k in range(n_omega):
                if u <= p_omega_cumsum[omega_j, k]:
                    new_j = k
                    break
            omega_j = new_j

        # ---- recording periods ----
        for t in range(n_months):
            p  = _interp_row(p_policy_T, omega_j, s, s_lo, inv_h, ns)
            n  = _interp_row(n_policy_T, omega_j, s, s_lo, inv_h, ns)
            nu = nu_draws[firm, burn_in + t]
            D  = min(nu * p ** neg_eps, s)

            inv_out[firm, t] = s
            dem_out[firm, t] = D
            rev_out[firm, t] = p * D
            exp_out[firm, t] = omega_grid[omega_j] * D ** gamma

            s = max((1.0 - delta) * (s - D + n), 0.0)

            # Markov transition for omega
            u     = omega_u_draws[firm, burn_in + t]
            new_j = n_omega - 1
            for k in range(n_omega):
                if u <= p_omega_cumsum[omega_j, k]:
                    new_j = k
                    break
            omega_j = new_j

    return inv_out, dem_out, rev_out, exp_out


# ---------------------------------------------------------------------------
# Python wrapper: simulate_firm
# ---------------------------------------------------------------------------

def simulate_firm(n_firms, n_months, p_policy, n_policy, params, seed=None, burn_in=100):
    """Simulate n_firms firms for n_months months each.

    Replicates ModelFunctions.jl::simulate_firm with vectorized pre-drawn
    randoms and a Numba-compiled parallel kernel over firms.

    Parameters
    ----------
    n_firms   : int
    n_months  : int
    p_policy  : (ns, n_omega) float64  — price policy from solve_value_function
    n_policy  : (ns, n_omega) float64  — order policy from solve_value_function
    params    : Parameters
    seed      : int or None
    burn_in   : int  (default 100)

    Returns
    -------
    inv_out, dem_out, rev_out, exp_out  — each (n_firms, n_months) float64
        inv_out : beginning-of-period inventory
        dem_out : realized demand
        rev_out : revenue  (p * D)
        exp_out : operating expense  (omega * D^gamma)
    """
    rng   = np.random.default_rng(seed)
    ns    = params.ns
    total = burn_in + n_months

    # Pre-draw all random numbers outside Numba
    nu_draws      = rng.lognormal(params.mu_log_nu, params.sigma_log_nu,
                                  size=(n_firms, total))
    omega_u_draws = rng.uniform(0.0, 1.0, size=(n_firms, total))
    omega_init_u  = rng.uniform(0.0, 1.0, size=n_firms)
    s_init_idx    = rng.integers(0, ns, size=n_firms)  # discrete uniform over grid

    # Precompute cumulative sums for inverse-CDF omega draws
    p_omega_cumsum  = np.ascontiguousarray(np.cumsum(params.p_omega, axis=1))  # (n_omega, n_omega)
    pi_omega_cumsum = np.ascontiguousarray(np.cumsum(params.pi_omega))         # (n_omega,)

    # Transpose policies to (n_omega, ns): thread j owns a contiguous row
    p_policy_T = np.ascontiguousarray(p_policy.T)
    n_policy_T = np.ascontiguousarray(n_policy.T)

    s_lo  = float(params.sgrid[0])
    inv_h = float((ns - 1) / (params.sgrid[-1] - params.sgrid[0]))

    return _simulate_firms(
        p_policy_T, n_policy_T,
        params.sgrid, s_lo, inv_h,
        params.omega_grid, p_omega_cumsum, pi_omega_cumsum,
        nu_draws, omega_u_draws, omega_init_u, s_init_idx,
        n_firms, n_months, burn_in,
        params.epsilon, params.delta, params.gamma,
    )


# ---------------------------------------------------------------------------
# Step 2: Moment computation
# ---------------------------------------------------------------------------

def estimate_omega_ar1(log_omega_proxy_panel):
    """OLS AR(1) on a balanced panel of log-omega proxies.

    Replicates EstimationFunctions.jl::estimate_omega_ar1, vectorized.

    Parameters
    ----------
    log_omega_proxy_panel : (n_firms, n_years) float64

    Returns
    -------
    mu_eta, sigma_eta2, rho_omega : floats
        mu_eta     = AR(1) intercept  (mean of innovation eta)
        sigma_eta2 = innovation variance (OLS residual variance)
        rho_omega  = AR(1) slope
    """
    # Within each firm, lag is valid from year 1 onward; no cross-firm contamination
    # because we slice within the firm axis.
    y = log_omega_proxy_panel[:, 1:].ravel()   # (n_firms * (n_years-1),)
    x = log_omega_proxy_panel[:, :-1].ravel()  # lagged values

    T     = len(y)
    x_bar = x.mean()
    y_bar = y.mean()
    Sxx   = np.dot(x - x_bar, x - x_bar)
    rho_omega  = np.dot(x - x_bar, y - y_bar) / Sxx
    mu_eta     = y_bar - rho_omega * x_bar
    resid      = y - (mu_eta + rho_omega * x)
    sigma_eta2 = np.dot(resid, resid) / (T - 2)

    return mu_eta, sigma_eta2, rho_omega


def compute_annual_auxiliary(tot_opex, tot_sales, tot_rev):
    """OLS gamma + AR(1) omega from annual aggregates.

    Replicates EstimationFunctions.jl::compute_annual_auxiliary, using
    NumPy instead of DataFrames.

    Parameters
    ----------
    tot_opex  : (n_firms, n_years) float64 — total annual operating expenses
    tot_sales : (n_firms, n_years) float64 — total annual sales quantity
    tot_rev   : (n_firms, n_years) float64 — total annual revenue

    Returns
    -------
    dict with keys: gamma_OLS, rho_omega, sigma_eta2, avg_opex_sales
    """
    log_opex  = np.log(tot_opex).ravel()
    log_sales = np.log(tot_sales).ravel()

    # OLS: log_opex = a + gamma * log_sales
    X               = np.column_stack([np.ones(len(log_opex)), log_sales])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_opex, rcond=None)
    a_ols, gamma_OLS = coeffs

    # log_omega_proxy = a + residuals = log_opex - gamma * log_sales
    log_omega_proxy = log_opex - gamma_OLS * log_sales  # (n_firms * n_years,)

    n_firms, n_years = tot_opex.shape
    log_omega_panel  = log_omega_proxy.reshape(n_firms, n_years)

    mu_eta, sigma_eta2, rho_omega = estimate_omega_ar1(log_omega_panel)

    valid_os = (tot_opex > 0.0) & (tot_rev > 0.0)
    avg_opex_sales = float((tot_opex[valid_os] / tot_rev[valid_os]).mean())

    return {"γ_OLS": gamma_OLS, "ρ_ω": rho_omega,
            "σ_η2": sigma_eta2, "avg_opex_sales": avg_opex_sales}


def compute_monthly_moments(inv_out, dem_out, rev_out, c):
    """Compute three monthly moments from simulation output arrays.

    Replicates EstimationFunctions.jl::compute_monthly_moments, vectorized.

    Parameters
    ----------
    inv_out : (n_firms, n_months) — beginning-of-period inventory
    dem_out : (n_firms, n_months) — realized demand
    rev_out : (n_firms, n_months) — revenue
    c       : float — marginal cost (params.c)

    Returns
    -------
    dict with keys: avg_isr, var_log1p_isr, avg_gross_margin
    """
    valid       = rev_out > 0.0
    isr         = inv_out[valid] / rev_out[valid]
    avg_isr          = float(isr.mean())
    var_log1p_isr    = float(np.log1p(isr).var())
    avg_gross_margin = float((rev_out[valid] / (c * dem_out[valid])).mean())

    return {"avg_isr": avg_isr, "var_log1p_isr": var_log1p_isr,
            "avg_gross_margin": avg_gross_margin}


# ---------------------------------------------------------------------------
# Step 3: _simulate_all_moments
# ---------------------------------------------------------------------------

def simulate_all_moments(params, p_policy, n_policy, n_firms, n_years, seed, burn_in=100):
    """Simulate panel and return all 7 estimation moments.

    Replicates EstimationFunctions.jl::_simulate_all_moments.

    Parameters
    ----------
    params    : Parameters
    p_policy  : (ns, n_omega) — price policy
    n_policy  : (ns, n_omega) — order policy
    n_firms   : int
    n_years   : int
    seed      : int
    burn_in   : int

    Returns
    -------
    dict with keys:
        avg_isr, var_log1p_isr, avg_gross_margin  (monthly)
        gamma_OLS, rho_omega, sigma_eta2, avg_opex_sales  (annual auxiliary)
    """
    n_months = n_years * 12

    inv_out, dem_out, rev_out, exp_out = simulate_firm(
        n_firms, n_months, p_policy, n_policy, params, seed=seed, burn_in=burn_in,
    )

    # Monthly moments
    mo = compute_monthly_moments(inv_out, dem_out, rev_out, params.c)

    # Annual aggregation: reshape (n_firms, n_years, 12) then sum over months
    tot_opex  = exp_out.reshape(n_firms, n_years, 12).sum(axis=2)   # (n_firms, n_years)
    tot_sales = dem_out.reshape(n_firms, n_years, 12).sum(axis=2)
    tot_rev   = rev_out.reshape(n_firms, n_years, 12).sum(axis=2)

    # Drop zero entries before log (shouldn't occur in practice)
    valid_ann = (tot_opex > 0.0) & (tot_sales > 0.0)
    ann = compute_annual_auxiliary(
        np.where(valid_ann, tot_opex,  1.0),
        np.where(valid_ann, tot_sales, 1.0),
        np.where(valid_ann, tot_rev,   1.0),
    )

    return {
        "avg_isr":          mo["avg_isr"],
        "var_log1p_isr":    mo["var_log1p_isr"],
        "avg_gross_margin": mo["avg_gross_margin"],
        "γ_OLS":            ann["γ_OLS"],
        "ρ_ω":          ann["ρ_ω"],
        "σ_η2":           ann["σ_η2"],
        "avg_opex_sales": ann["avg_opex_sales"],
    }


# ---------------------------------------------------------------------------
# Step 4: Parameter packing / unpacking (matches Julia estimate_params_ii_full)
# Parameter order: [gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta]
# ---------------------------------------------------------------------------

# Bounds (used by both pack and unpack)
# Must match simulate_moments_main.py / SimulateMoments.jl
_BOUNDS = {
    "gamma":      (0.5,              1.25),
    "mu_eta":     (math.log(0.0001), math.log(0.5)),
    "sigma_eta2": (0.025,            0.15),
    "rho_omega":  (0.0,              0.9),
    "sigma_nu2":  (0.01,             0.3),
    "epsilon":    (4.0,              20.0),
    "delta":      (0.005,            0.1),
}


def _pack(theta):
    """Map structural params to unconstrained search vector.

    theta : [gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta]
    returns: [gamma, mu_eta, log(sigma_eta2), arctanh(rho_omega),
              log(sigma_nu2), epsilon, logit(delta)]
    """
    gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta = theta
    import math
    return np.array([
        gamma,
        mu_eta,
        math.log(sigma_eta2),
        math.atanh(rho_omega),
        math.log(sigma_nu2),
        epsilon,
        math.log(delta / (1.0 - delta)),
    ])


def _unpack(x):
    """Map unconstrained search vector back to structural params with clamping.

    Inverse of _pack.
    Returns list: [gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta]
    """
    import math
    b = _BOUNDS
    gamma      = float(np.clip(x[0], *b["gamma"]))
    mu_eta     = float(np.clip(x[1], *b["mu_eta"]))
    sigma_eta2 = float(np.clip(math.exp(x[2]),           *b["sigma_eta2"]))
    rho_omega  = float(np.clip(math.tanh(x[3]),          *b["rho_omega"]))
    sigma_nu2  = float(np.clip(math.exp(x[4]),           *b["sigma_nu2"]))
    epsilon    = float(np.clip(x[5],                     *b["epsilon"]))
    delta      = float(np.clip(1.0 / (1.0 + math.exp(-x[6])), *b["delta"]))
    return [gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta]


# ---------------------------------------------------------------------------
# Step 5: estimate_params_ii_full
# ---------------------------------------------------------------------------

def estimate_params_ii_full(
    target_moments,
    init_guess,
    W,
    params_base=None,
    n_firms=200,
    n_years=50,
    seed=212311,
    max_iter=1000,
    verbose=True,
    simplex_scale=0.15,
    n_restarts=1,
    method="nelder-mead",
    tol=1e-4,
):
    """Indirect inference estimator for 7 structural parameters.

    Replicates EstimationFunctions.jl::estimate_params_ii_full.

    Parameters
    ----------
    target_moments : dict with keys avg_isr, var_log1p_isr, avg_gross_margin,
                     gamma_OLS, rho_omega, sigma_eta2, mu_eta
    init_guess     : length-7 list/array [gamma, mu_eta, sigma_eta2, rho_omega,
                     sigma_nu2, epsilon, delta]
    W              : (7, 7) weighting matrix
    params_base    : Parameters to inherit fixed fields (c, fc, beta, ns, size);
                     uses Parameters() defaults if None
    n_firms        : int
    n_years        : int
    seed           : int
    max_iter       : int
    verbose        : bool

    Returns
    -------
    dict with keys: gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon,
                    delta, obj_value, result (scipy OptimizeResult)
    """
    from scipy.optimize import minimize
    import math

    if params_base is None:
        params_base = Parameters()

    m_hat = np.array([
        target_moments["avg_isr"],
        target_moments["var_log1p_isr"],
        target_moments["avg_gross_margin"],
        target_moments["γ_OLS"],
        target_moments["ρ_ω"],
        target_moments["σ_η2"],
        target_moments["avg_opex_sales"],
    ])

    method_key = str(method).strip().lower()
    valid_methods = {"nelder-mead", "bobyqa", "cma-es"}
    if method_key not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of {sorted(valid_methods)}"
        )

    iter_count = [0]

    def objective(x):
        iter_count[0] += 1
        if method_key == "bobyqa":
            gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta = (
                float(x[0]), float(x[1]), float(x[2]), float(x[3]),
                float(x[4]), float(x[5]), float(x[6]),
            )
        else:
            gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta = _unpack(x)
        try:
            p = Parameters(
                c=params_base.c, fc=params_base.fc,
                mu_eta=mu_eta, sigma_eta2=sigma_eta2, rho_omega=rho_omega,
                gamma=gamma, delta=delta, beta=params_base.beta,
                epsilon=epsilon, mu_nu=params_base.mu_nu, sigma_nu2=sigma_nu2,
                ns=params_base.ns, size=params_base.size,
            )
            sol    = solve_value_function(p)
            m_tilde_d = simulate_all_moments(
                p, sol["p_policy"], sol["n_policy"], n_firms, n_years, seed,
            )
            m_tilde = np.array([
                m_tilde_d["avg_isr"],
                m_tilde_d["var_log1p_isr"],
                m_tilde_d["avg_gross_margin"],
                m_tilde_d["γ_OLS"],
                m_tilde_d["ρ_ω"],
                m_tilde_d["σ_η2"],
                m_tilde_d["avg_opex_sales"],
            ])
            M   = m_hat - m_tilde
            sse = float(M @ W @ M)
            if verbose:
                print(
                    f"  {iter_count[0]:4d} | "
                    f"{m_tilde[0]:7.4f} | {m_tilde[1]:9.4f} | {m_tilde[2]:7.4f} | "
                    f"{m_tilde[3]:7.4f} | {m_tilde[4]:7.4f} | {m_tilde[5]:7.4f} | "
                    f"{m_tilde[6]:7.4f} | {sse:9.5f}"
                )
            return sse
        except Exception:
            return 1e10

    if method_key == "bobyqa":
        x0 = np.array(init_guess, dtype=np.float64)
    else:
        x0 = _pack(list(init_guess))

    def _make_simplex(x, scale):
        """Build an (n+1, n) initial simplex by perturbing each coordinate."""
        n = len(x)
        simplex = np.empty((n + 1, n))
        simplex[0] = x
        for j in range(n):
            row = x.copy()
            row[j] += scale * max(abs(x[j]), 1.0)
            simplex[j + 1] = row
        return simplex

    if verbose:
        print("\n=== Full II Estimation — Data Moments ===")
        keys = ["avg_isr","var_log1p_isr","avg_gross_margin","γ_OLS","ρ_ω","σ_η2","avg_opex_sales"]
        for k, v in zip(keys, m_hat):
            print(f"  {k:20s} = {v:10.6f}")
        print(f"\nOptimizer: {method_key}, max_iter={max_iter}, n_restarts={n_restarts}")
        print("\n iter | avg_isr | var_log1p | avg_gm  | gamma_OLS| rho_w   | sig_eta2 | opex_sales  | obj")
        print("-" * 95)

    x_best     = x0
    obj_best   = np.inf
    opt_result = None

    for restart in range(n_restarts + 1):
        if verbose and restart > 0:
            print(f"\n--- Warm restart {restart} (best obj so far: {obj_best:.6f}) ---")

        if method_key == "nelder-mead":
            simplex = _make_simplex(x_best, simplex_scale)
            res = minimize(
                objective, x_best, method="Nelder-Mead",
                options={
                    "maxiter": max_iter,
                    "xatol": tol,
                    "fatol": tol,
                    "disp": False,
                    "initial_simplex": simplex,
                },
            )

        elif method_key == "bobyqa":
            import pybobyqa

            bounds = np.array([
                _BOUNDS["gamma"],
                _BOUNDS["mu_eta"],
                _BOUNDS["sigma_eta2"],
                _BOUNDS["rho_omega"],
                _BOUNDS["sigma_nu2"],
                _BOUNDS["epsilon"],
                _BOUNDS["delta"],
            ], dtype=np.float64)

            lb, ub = bounds[:, 0], bounds[:, 1]
            rhobeg = 0.1 * float(np.min(ub - lb))
            # x0 must be strictly interior by at least rhobeg on every coordinate
            x_bobyqa = np.clip(x_best, lb + rhobeg, ub - rhobeg)

            res_bobyqa = pybobyqa.solve(
                objective,
                x0=x_bobyqa,
                bounds=(lb, ub),
                rhobeg=rhobeg,
                rhoend=tol,
                maxfun=max_iter,
                seek_global_minimum=False,
                print_progress=False,
            )

            class _BobyqaResult:
                pass

            res = _BobyqaResult()
            res.x = np.array(res_bobyqa.x, dtype=np.float64)
            res.fun = float(res_bobyqa.f)
            res.success = bool(res_bobyqa.flag > 0)

        else:  # method_key == "cma-es"
            import cma

            bounds = [
                [
                    _BOUNDS["gamma"][0],
                    _BOUNDS["mu_eta"][0],
                    np.log(_BOUNDS["sigma_eta2"][0]),
                    np.arctanh(_BOUNDS["rho_omega"][0]),
                    np.log(_BOUNDS["sigma_nu2"][0]),
                    _BOUNDS["epsilon"][0],
                    np.log(_BOUNDS["delta"][0] / (1.0 - _BOUNDS["delta"][0])),
                ],
                [
                    _BOUNDS["gamma"][1],
                    _BOUNDS["mu_eta"][1],
                    np.log(_BOUNDS["sigma_eta2"][1]),
                    np.arctanh(_BOUNDS["rho_omega"][1]),
                    np.log(_BOUNDS["sigma_nu2"][1]),
                    _BOUNDS["epsilon"][1],
                    np.log(_BOUNDS["delta"][1] / (1.0 - _BOUNDS["delta"][1])),
                ],
            ]

            sigma0 = max(simplex_scale, 1e-2)
            opts = {
                "maxfevals": max_iter,
                "verbose": -9 if not verbose else 1,
                "bounds": bounds,
            }
            es = cma.CMAEvolutionStrategy(np.array(x_best, dtype=np.float64), sigma0, opts)
            es.optimize(objective)

            class _CmaResult:
                pass

            res = _CmaResult()
            res.x = np.array(es.result.xbest, dtype=np.float64)
            res.fun = float(es.result.fbest)
            res.success = bool(np.isfinite(res.fun))

        if res.fun < obj_best:
            obj_best   = res.fun
            x_best     = res.x
            opt_result = res

    if method_key == "bobyqa":
        gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta = (
            float(opt_result.x[0]), float(opt_result.x[1]), float(opt_result.x[2]),
            float(opt_result.x[3]), float(opt_result.x[4]), float(opt_result.x[5]),
            float(opt_result.x[6]),
        )
    else:
        gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta = _unpack(opt_result.x)

    if verbose:
        print("\n=== Full II Estimation Complete ===")
        print(f"  Converged  : {opt_result.success}")
        print(f"  gamma      = {gamma:.6f}")
        print(f"  mu_eta     = {mu_eta:.6f}")
        print(f"  sigma_eta2 = {sigma_eta2:.6f}")
        print(f"  rho_omega  = {rho_omega:.6f}")
        print(f"  sigma_nu2  = {sigma_nu2:.6f}")
        print(f"  epsilon    = {epsilon:.6f}")
        print(f"  delta      = {delta:.6f}")
        print(f"  Objective  : {opt_result.fun:.8f}")

    return {
        "gamma":      gamma,
        "mu_eta":     mu_eta,
        "sigma_eta2": sigma_eta2,
        "rho_omega":  rho_omega,
        "sigma_nu2":  sigma_nu2,
        "epsilon":    epsilon,
        "delta":      delta,
        "obj_value":  float(opt_result.fun),
        "result":     opt_result,
    }


# ---------------------------------------------------------------------------
# Step 6: select_best_grid_start
# ---------------------------------------------------------------------------

def select_best_grid_start(df_grid, target_moments, W):
    """Select the parameter vector from a precomputed grid that minimises the
    weighted II objective against target_moments.

    Replicates EstimationFunctions.jl::select_best_grid_start, vectorized.

    Parameters
    ----------
    df_grid        : pandas DataFrame (or path str/Path to CSV) with columns
                     gamma, mu_eta (or mue_eta), sigma_eta2, rho_omega,
                     sigma_nu2 (or signu2), epsilon, delta, avg_isr,
                     var_log1p_isr, avg_gross_margin, gamma_OLS, rho_omega
                     (col), sigma_eta2 (col), mu_eta (col), failed
    target_moments : dict — same keys as simulate_all_moments output
    W              : (7, 7) weighting matrix

    Returns
    -------
    dict with keys: row_index, obj_value, gamma, mu_eta, sigma_eta2,
                    rho_omega, sigma_nu2, epsilon, delta
    """
    import pandas as pd

    if not isinstance(df_grid, pd.DataFrame):
        df_grid = pd.read_csv(df_grid)

    m_hat = np.array([
        target_moments["avg_isr"],
        target_moments["var_log1p_isr"],
        target_moments["avg_gross_margin"],
        target_moments["γ_OLS"],
        target_moments["ρ_ω"],
        target_moments["σ_η2"],
        target_moments["avg_opex_sales"],
    ])

    ok = ~df_grid["failed"].astype(bool)
    moment_cols = ["avg_isr", "var_log1p_isr", "avg_gross_margin",
                   "γ_OLS", "ρ_ω", "σ_η2", "avg_opex_sales"]
    ok = ok & ~df_grid[moment_cols].isna().any(axis=1)
    ok = ok & ~np.isinf(df_grid[moment_cols].to_numpy(dtype=np.float64)).any(axis=1)
    M_all = df_grid.loc[ok, moment_cols].to_numpy(dtype=np.float64)  # (n_ok, 7)

    # Vectorised objective: each row is m_hat - m_tilde
    diff    = m_hat[None, :] - M_all                  # (n_ok, 7)
    obj_all = np.einsum("ni,ij,nj->n", diff, W, diff) # (n_ok,)

    best_local = int(np.argmin(obj_all))
    best_global = df_grid.index[ok][best_local]

    row = df_grid.loc[best_global]
    return {
        "row_index": best_global,
        "obj_value": float(obj_all[best_local]),
        "gamma":      float(row["γ"]),
        "mu_eta":     float(row["μη"]),
        "sigma_eta2": float(row["ση2"]),
        "rho_omega":  float(row["ρω"]),
        "sigma_nu2":  float(row["σν2"]),
        "epsilon":    float(row["ϵ"]),
        "delta":      float(row["δ"]),
    }


# ---------------------------------------------------------------------------
# Stage 3 helper: parameter sweep moments on grid
# ---------------------------------------------------------------------------

def _grid_worker(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    import numba as _nb

    idx, theta_row, n_firms, n_years, seed, threads_per_worker = args
    _nb.set_num_threads(threads_per_worker)

    gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta = theta_row
    row = {
        "γ": float(gamma), "μη": float(mu_eta), "ση2": float(sigma_eta2),
        "ρω": float(rho_omega), "σν2": float(sigma_nu2),
        "ϵ": float(epsilon), "δ": float(delta),
        "avg_isr": np.nan, "var_log1p_isr": np.nan,
        "avg_gross_margin": np.nan, "γ_OLS": np.nan,
        "ρ_ω": np.nan, "σ_η2": np.nan, "avg_opex_sales": np.nan,
        "any_inventory_above_grid": False, "failed": True,
    }
    try:
        p = Parameters(
            c=1.0, fc=0.0,
            mu_eta=float(mu_eta), sigma_eta2=float(sigma_eta2),
            rho_omega=float(rho_omega), gamma=float(gamma),
            delta=float(delta), beta=0.95,
            epsilon=float(epsilon), mu_nu=1.0, sigma_nu2=float(sigma_nu2),
            ns=200, scale=1.0, size=100.0,
        )
        sol = solve_value_function(p)
        m = simulate_all_moments(
            p, sol["p_policy"], sol["n_policy"],
            n_firms=n_firms, n_years=n_years, seed=seed,
        )
        row.update({
            "avg_isr": float(m["avg_isr"]),
            "var_log1p_isr": float(m["var_log1p_isr"]),
            "avg_gross_margin": float(m["avg_gross_margin"]),
            "γ_OLS": float(m["γ_OLS"]),
            "ρ_ω": float(m["ρ_ω"]),
            "σ_η2": float(m["σ_η2"]),
            "avg_opex_sales": float(m["avg_opex_sales"]),
            "any_inventory_above_grid": False,
            "failed": False,
        })
    except Exception:
        pass
    return idx, row


def compute_moments_on_grid(param_vectors, n_firms=500, n_years=20, seed=212311,
                            output_path=None, verbose=True,
                            n_workers=None, threads_per_worker=1):
    """Compute simulated moments for a list/array of parameter vectors.

    Parameters
    ----------
    param_vectors : array-like, shape (n, 7)
        Parameter order: [gamma, mu_eta, sigma_eta2, rho_omega,
        sigma_nu2, epsilon, delta].
    n_firms : int
    n_years : int
    seed : int
    output_path : str/Path or None
        If provided, writes the output DataFrame to CSV.
    verbose : bool
    n_workers : int or None
        Number of worker processes.  None → use os.cpu_count() (all logical
        cores).  Set to 1 to stay single-process (no subprocess overhead).
    threads_per_worker : int
        Numba threads per worker process.  Default 1.  The product
        n_workers * threads_per_worker should not exceed the available
        logical cores to avoid over-subscription.

    Returns
    -------
    pd.DataFrame with columns:
      γ, μη, ση2, ρω, σν2, ϵ, δ,
      avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2,
      avg_opex_sales, any_inventory_above_grid, failed
    """
    import os
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor

    theta = np.asarray(param_vectors, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 7:
        raise ValueError("param_vectors must have shape (n, 7)")

    n_total = theta.shape[0]

    # Determine actual worker count
    max_available = os.cpu_count() or 1
    if n_workers is None:
        n_workers = max_available
    n_workers = max(1, min(n_workers, max_available, n_total))

    if verbose:
        print(f"  Workers: {n_workers}  threads_per_worker: {threads_per_worker}"
              f"  (logical cores available: {max_available})")

    jobs = [
        (i, theta[i], n_firms, n_years, seed, threads_per_worker)
        for i in range(n_total)
    ]

    if n_workers == 1:
        # Serial path — avoid ProcessPoolExecutor overhead
        results = []
        for job in jobs:
            results.append(_grid_worker(job))
            i = job[0]
            if verbose and ((i + 1) % 50 == 0 or i + 1 == n_total):
                n_ok = sum(1 for _, r in results if not r["failed"])
                done = i + 1
                print(f"  [{done:4d}/{n_total}] complete  "
                      f"(success={n_ok}, failed={done - n_ok})")
    else:
        # Parallel path
        results_unordered = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, row in pool.map(_grid_worker, jobs):
                results_unordered.append((i, row))
                if verbose and (len(results_unordered) % 50 == 0
                                or len(results_unordered) == n_total):
                    n_ok = sum(1 for _, r in results_unordered if not r["failed"])
                    done = len(results_unordered)
                    print(f"  [{done:4d}/{n_total}] complete  "
                          f"(success={n_ok}, failed={done - n_ok})")
        # Restore input order
        results = sorted(results_unordered, key=lambda x: x[0])

    rows = [row for _, row in results]

    df_out = pd.DataFrame(rows)

    if output_path is not None:
        df_out.to_csv(output_path, index=False)

    return df_out


# ---------------------------------------------------------------------------
# Step 7: Asymptotic variance of II estimates
# ---------------------------------------------------------------------------

_PARAM_NAMES = ["gamma", "mu_eta", "sigma_eta2", "rho_omega", "sigma_nu2",
                "epsilon", "delta"]


def _moment_vector_from_params(params, n_firms, n_years, seed, maxiter=1000):
    """Solve VFI and simulate moments; return length-7 array in standard order."""
    sol = solve_value_function(params, maxiter=maxiter)
    if not sol["converged"]:
        raise RuntimeError("solve_value_function did not converge")
    m = simulate_all_moments(
        params, sol["p_policy"], sol["n_policy"], n_firms, n_years, seed,
    )
    return np.array([
        m["avg_isr"], m["var_log1p_isr"], m["avg_gross_margin"],
        m["γ_OLS"],   m["ρ_ω"],          m["σ_η2"],            m["avg_opex_sales"],
    ])


def _params_from_theta(params_base, theta):
    """Build a Parameters object from length-7 vector [γ, μη, ση2, ρω, σν2, ϵ, δ]."""
    gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta = theta
    return Parameters(
        c=params_base.c, fc=params_base.fc, beta=params_base.beta,
        mu_nu=params_base.mu_nu, ns=params_base.ns, size=params_base.size,
        gamma=gamma, mu_eta=mu_eta, sigma_eta2=sigma_eta2, rho_omega=rho_omega,
        sigma_nu2=sigma_nu2, epsilon=epsilon, delta=delta,
    )


def compute_ii_jacobian(params_hat, n_firms=5000, n_years=20, seed=212311,
                        maxiter=1000):
    """Numerically compute the 7×7 Jacobian of simulated moments w.r.t. parameters.

    Replicates EstimationFunctions.jl::compute_full_ii_jacobian.
    Uses central differences with step sizes matching the Julia implementation.

    Parameter order: [γ, μη, ση2, ρω, σν2, ϵ, δ]
    Moment order:    [avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, avg_opex_sales]

    Returns
    -------
    G : (7, 7) ndarray — G[i, j] = ∂m_i / ∂θ_j
    """
    theta0 = np.array([
        params_hat.gamma, params_hat.mu_eta, params_hat.sigma_eta2,
        params_hat.rho_omega, params_hat.sigma_nu2, params_hat.epsilon,
        params_hat.delta,
    ])

    # Step sizes: match Julia's max(|θ_j| * 1e-4, 1e-6) with larger floors for ρω and δ
    step = np.maximum(np.abs(theta0) * 1e-4, 1e-6)
    step[3] = max(step[3], 1e-5)  # ρ_ω
    step[6] = max(step[6], 1e-5)  # δ

    G = np.empty((7, 7))
    for j in range(7):
        t_plus  = theta0.copy(); t_plus[j]  += step[j]
        t_minus = theta0.copy(); t_minus[j] -= step[j]
        m_plus  = _moment_vector_from_params(
            _params_from_theta(params_hat, t_plus),  n_firms, n_years, seed, maxiter)
        m_minus = _moment_vector_from_params(
            _params_from_theta(params_hat, t_minus), n_firms, n_years, seed, maxiter)
        G[:, j] = (m_plus - m_minus) / (2.0 * step[j])

    return G


def compute_ii_asymptotic_variance(params_hat, W, n_firms=5000, n_years=20,
                                   seed=212311, sample_size=1, maxiter=1000,
                                   verbose=True):
    """Compute the efficient-GMM asymptotic variance of the II estimator.

    Replicates EstimationFunctions.jl::compute_full_ii_asymptotic_variance.

    Numerically evaluates the Jacobian G at params_hat, then applies:
        Avar(θ̂) = (G W G')⁻¹
        Vcov(θ̂) = Avar(θ̂) / sample_size

    Parameters
    ----------
    params_hat  : Parameters at the estimated point
    W           : (7, 7) weighting matrix (inverse of moment covariance)
    n_firms     : int  — firms used for Jacobian simulation (default 5000)
    n_years     : int
    seed        : int
    sample_size : int  — divides Avar to give finite-sample Vcov
    maxiter     : int
    verbose     : bool

    Returns
    -------
    dict with keys: G, avar, vcov, se
        G    : (7, 7) Jacobian
        avar : (7, 7) asymptotic variance matrix
        vcov : (7, 7) finite-sample covariance (avar / sample_size)
        se   : (7,)  standard errors (sqrt of vcov diagonal)
    """
    if verbose:
        print("Computing Jacobian (14 VFI + simulation calls)...")

    G    = compute_ii_jacobian(params_hat, n_firms=n_firms, n_years=n_years,
                               seed=seed, maxiter=maxiter)
    avar = np.linalg.inv(G @ W @ G.T)
    vcov = avar / sample_size
    se   = np.sqrt(np.diag(vcov))

    if verbose:
        print("\nAsymptotic standard errors:")
        for name, s in zip(_PARAM_NAMES, se):
            print(f"  {name:12s}  SE = {s:.6f}")

    return {"G": G, "avar": avar, "vcov": vcov, "se": se}
