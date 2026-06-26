import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numba import jit, prange
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar, brentq
from scipy.special import ndtri


SQRT_PI = math.sqrt(math.pi)
NORMAL = NormalDist()


@dataclass
class UniformInterp:
    values: np.ndarray
    s_lo: float
    inv_h: float

    def __call__(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        t = (x_arr - self.s_lo) * self.inv_h
        n = self.values.shape[0]

        out = np.empty_like(t, dtype=np.float64)

        lower = t <= 0.0
        upper = t >= (n - 1)
        middle = (~lower) & (~upper)

        if np.any(lower):
            out[lower] = self.values[0] + t[lower] * (self.values[1] - self.values[0])

        if np.any(upper):
            excess = t[upper] - (n - 1)
            out[upper] = self.values[-1] + excess * (self.values[-1] - self.values[-2])

        if np.any(middle):
            t_mid = t[middle]
            i0 = np.floor(t_mid).astype(np.int64)
            alpha = t_mid - i0
            out[middle] = self.values[i0] + alpha * (self.values[i0 + 1] - self.values[i0])

        if np.isscalar(x):
            return float(out)
        return out


class Parameters:
    def __init__(
        self,
        c=1.0,
        fc=0.0,
        mu_eta=0.0,
        sigma_eta2=0.0,
        rho_omega=0.9,
        gamma=1.0,
        delta=0.2,
        epsilon=2.0,
        sigma_nu2=0.15,
        mu_nu=1.0,
        beta=0.995,
        q=19,
        q_omega=7,
        scale=1.0,
        size=1.0,
        ns=400,
    ):
        x, w = hermgauss(q)
        gl_x, gl_w = leggauss(q)

        scale_parameter = scale ** epsilon
        c_adj = c * scale
        mu_nu_adj = mu_nu * scale_parameter * size
        sigma_nu2_adj = sigma_nu2 * (scale_parameter ** 2) * (size ** 2)
        mu_eta_adj = mu_eta + (1.0 - rho_omega) * math.log(scale) + (1.0 - rho_omega) * math.log(size ** (1.0 - gamma))

        sigma2 = math.log(1.0 + sigma_nu2_adj / (mu_nu_adj ** 2))
        sigma = math.sqrt(sigma2)
        mu = math.log(mu_nu_adj) - 0.5 * sigma2

        x_lognormal = np.exp(mu + math.sqrt(2.0) * sigma * x)

        mu_log_omega = mu_eta_adj / (1.0 - rho_omega) if abs(1.0 - rho_omega) > 1e-10 else mu_eta_adj
        sigma_eta = math.sqrt(max(sigma_eta2, 0.0))

        if sigma_eta2 <= 0.0 or q_omega <= 1:
            omega_grid = np.array([math.exp(mu_log_omega)], dtype=np.float64)
            p_omega = np.ones((1, 1), dtype=np.float64)
            pi_omega = np.array([1.0], dtype=np.float64)
        else:
            sigma_log_omega = sigma_eta / math.sqrt(1.0 - rho_omega ** 2)
            m_tau = 3.0
            log_omega_lo = mu_log_omega - m_tau * sigma_log_omega
            log_omega_hi = mu_log_omega + m_tau * sigma_log_omega
            log_omega_grid = np.linspace(log_omega_lo, log_omega_hi, q_omega)
            h = log_omega_grid[1] - log_omega_grid[0]
            omega_grid = np.exp(log_omega_grid)

            p_omega = np.zeros((q_omega, q_omega), dtype=np.float64)
            for i in range(q_omega):
                cond_mean = mu_log_omega + rho_omega * (log_omega_grid[i] - mu_log_omega)
                for j in range(q_omega):
                    if j == 0:
                        p_omega[i, j] = NORMAL.cdf((log_omega_grid[j] + 0.5 * h - cond_mean) / sigma_eta)
                    elif j == (q_omega - 1):
                        p_omega[i, j] = 1.0 - NORMAL.cdf((log_omega_grid[j] - 0.5 * h - cond_mean) / sigma_eta)
                    else:
                        z_hi = (log_omega_grid[j] + 0.5 * h - cond_mean) / sigma_eta
                        z_lo = (log_omega_grid[j] - 0.5 * h - cond_mean) / sigma_eta
                        p_omega[i, j] = NORMAL.cdf(z_hi) - NORMAL.cdf(z_lo)
                p_omega[i, :] /= p_omega[i, :].sum()

            pi_omega = np.full(q_omega, 1.0 / q_omega, dtype=np.float64)
            for _ in range(2000):
                pi_omega = p_omega.T @ pi_omega
            pi_omega /= pi_omega.sum()

        mu_nu_back = mu_nu_adj / (scale_parameter * size)
        sigma_nu2_back = sigma_nu2_adj / ((scale_parameter ** 2) * (size ** 2))

        lim_p = limit_price(mu, sigma, epsilon, gamma, c_adj, omega_grid[0])
        smax = 1.5*math.exp(mu + sigma * NORMAL.inv_cdf(0.95)) * lim_p ** (-epsilon)
        sgrid = np.linspace(0.0, smax, ns)

        self.c = float(c_adj)
        self.fc = float(fc)
        self.mu_eta = float(mu_eta_adj)
        self.sigma_eta2 = float(sigma_eta2)
        self.rho_omega = float(rho_omega)
        self.q_omega = int(omega_grid.shape[0])
        self.omega_grid = omega_grid
        self.p_omega = p_omega
        self.pi_omega = pi_omega
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.mu_nu = float(mu_nu_back)
        self.sigma_nu2 = float(sigma_nu2_back)
        self.mu_log_nu = float(mu)
        self.sigma_log_nu = float(sigma)
        self.q = int(q)
        self.quad_nodes = x.astype(np.float64)
        self.quad_weights = w.astype(np.float64)
        self.quad_nodes_lognormal = x_lognormal.astype(np.float64)
        self.gl_nodes = gl_x.astype(np.float64)
        self.gl_weights = gl_w.astype(np.float64)
        self.smax = float(smax)
        self.ns = int(ns)
        self.sgrid = sgrid.astype(np.float64)
        self.size = float(size)


def lognormal_cdf(x, mu, sigma):
    if x <= 0.0:
        return 0.0
    z = (math.log(x) - mu) / sigma
    return NORMAL.cdf(z)


def truncated_lognormal_mean(nu_bar, params):
    mu_log = params.mu_log_nu
    sigma_log = params.sigma_log_nu

    z1 = (math.log(nu_bar) - mu_log - sigma_log * sigma_log) / sigma_log
    z0 = (math.log(nu_bar) - mu_log) / sigma_log

    return math.exp(mu_log + 0.5 * sigma_log * sigma_log) * NORMAL.cdf(z1) / NORMAL.cdf(z0)


def truncated_lognormal_ratio_e_nu_gamma_over_e_nu(nu_bar, params):
    f_bar = lognormal_cdf(nu_bar, params.mu_log_nu, params.sigma_log_nu)
    if f_bar <= 0.0:
        return 0.0, 0.0, 0.0

    half_fbar = 0.5 * f_bar
    u_q = np.clip(half_fbar * (params.gl_nodes + 1.0), 1e-12, 1.0 - 1e-12)
    nu_q = np.exp(params.mu_log_nu + params.sigma_log_nu * ndtri(u_q))
    num = float(params.gl_weights @ (nu_q ** params.gamma))
    den = float(params.gl_weights @ nu_q)

    ratio = (num / den) if den > 0.0 else 0.0
    return ratio, 0.5 * num, 0.5 * den


def price_residual(p, s, c_tilde, omega, params):
    nu_bar = s * (p ** params.epsilon)
    f_bar = lognormal_cdf(nu_bar, params.mu_log_nu, params.sigma_log_nu)
    tail = 1.0 - f_bar

    if f_bar < 1e-10:
        return 1e6

    e_nu = truncated_lognormal_mean(nu_bar, params)
    ratio_e_nu_gamma = truncated_lognormal_ratio_e_nu_gamma_over_e_nu(nu_bar, params)[0]

    opp_mc = omega * params.gamma * ratio_e_nu_gamma * (p ** (params.epsilon * (1.0 - params.gamma)))
    rhs = (
        (params.epsilon / (params.epsilon - 1.0)) * (opp_mc + c_tilde)
        + (1.0 / (params.epsilon - 1.0)) * s * (p ** (params.epsilon + 1.0)) * (1.0 / e_nu) * (tail / f_bar)
    )
    return p - rhs


def proxy_profit_unconst(p, c_tilde, omega, mu_log, sigma_log, epsilon, gamma):
    e_nu = math.exp(mu_log + 0.5 * sigma_log ** 2)
    e_nu_gamma = math.exp(gamma * mu_log + 0.5 * (gamma ** 2) * sigma_log ** 2)
    return e_nu * p ** (-epsilon) * (p - c_tilde) - omega * e_nu_gamma * p ** (-gamma * epsilon)


def proxy_profit(p, s, c_tilde, omega, params):
    nu_bar = s * p ** params.epsilon
    f_bar = lognormal_cdf(nu_bar, params.mu_log_nu, params.sigma_log_nu)
    tail = 1.0 - f_bar
    _, e_nu_gamma_trunc, e_nu_trunc = truncated_lognormal_ratio_e_nu_gamma_over_e_nu(nu_bar, params)
    e_d = p ** (-params.epsilon) * f_bar * e_nu_trunc + s * tail
    e_d_gamma = p ** (-params.gamma * params.epsilon) * f_bar * e_nu_gamma_trunc + (s ** params.gamma) * tail
    return (p - c_tilde) * e_d - omega * e_d_gamma


def neg_profit_check(mu_log, sigma_log, epsilon, gamma, c_tilde, omega):
    e_nu = math.exp(mu_log + 0.5 * sigma_log ** 2)
    e_nu_gamma = math.exp(gamma * mu_log + 0.5 * (gamma ** 2) * sigma_log ** 2)
    A = e_nu
    B = omega * e_nu_gamma
    a = epsilon * (1.0 - gamma)
    if c_tilde <= 0.0 or a < 1.0:
        return False
    if abs(a - 1.0) < 1e-10:
        return A <= B
    b_crit = A * ((a - 1.0) ** (a - 1.0)) / ((a ** a) * (c_tilde ** (a - 1.0)))
    return B >= b_crit


def limit_price(mu_log, sigma_log, epsilon, gamma, c_tilde, omega):
    obj = lambda p: -proxy_profit_unconst(p, c_tilde, omega, mu_log, sigma_log, epsilon, gamma)
    result = minimize_scalar(obj, bounds=(1e-3, 5.0), method='bounded')
    return result.x


@jit(nopython=True, fastmath=True, cache=True)
def _ev_single(n, s, d_col, c_col, p, vinterp_vals, s_lo, inv_h, fc, c_param, delta, beta, quad_weights):
    """Compiled EV evaluation for a single order quantity n."""
    n_pts = len(vinterp_vals)
    one_minus_delta = 1.0 - delta
    order_cost = (fc + c_param * n) if n > 0.0 else 0.0
    ev = 0.0
    for q in range(len(quad_weights)):
        s_tilde = s - d_col[q] + n
        t = (one_minus_delta * s_tilde - s_lo) * inv_h
        i0 = int(t)
        if t <= 0.0:
            cont = vinterp_vals[0] + t * (vinterp_vals[1] - vinterp_vals[0])
        elif i0 >= n_pts - 1:
            excess = t - (n_pts - 1)
            cont = vinterp_vals[n_pts - 1] + excess * (vinterp_vals[n_pts - 1] - vinterp_vals[n_pts - 2])
        else:
            alpha = t - i0
            cont = vinterp_vals[i0] + alpha * (vinterp_vals[i0 + 1] - vinterp_vals[i0])
        ev += quad_weights[q] * (p * d_col[q] - c_col[q] - order_cost + beta * cont)
    return ev / 1.7724538509055159


@jit(nopython=True, fastmath=True, cache=True)
def _optimize_n(s, d_col, c_col, p, vinterp_vals, s_lo, inv_h, fc, c_param, delta, beta, quad_weights, n_upper):
    """Compiled golden section over order quantity n. No Python callbacks."""
    invphi  = 0.6180339887498949   # (sqrt(5)-1)/2
    invphi2 = 0.3819660112501051   # (3-sqrt(5))/2
    n_iter  = 30                   # gives tol ~ 5e-6 for n_upper ~ 10; well within VFI tolerance

    a = 0.0
    b = n_upper
    h = b - a

    if h <= 1e-10:
        return 0.0, _ev_single(0.0, s, d_col, c_col, p, vinterp_vals, s_lo, inv_h, fc, c_param, delta, beta, quad_weights)

    c_pt = a + invphi2 * h
    d_pt = a + invphi  * h
    yc = -_ev_single(c_pt, s, d_col, c_col, p, vinterp_vals, s_lo, inv_h, fc, c_param, delta, beta, quad_weights)
    yd = -_ev_single(d_pt, s, d_col, c_col, p, vinterp_vals, s_lo, inv_h, fc, c_param, delta, beta, quad_weights)

    for _ in range(n_iter):
        if yc < yd:
            b    = d_pt
            d_pt = c_pt
            yd   = yc
            h    = invphi * h
            c_pt = a + invphi2 * h
            yc   = -_ev_single(c_pt, s, d_col, c_col, p, vinterp_vals, s_lo, inv_h, fc, c_param, delta, beta, quad_weights)
        else:
            a    = c_pt
            c_pt = d_pt
            yc   = yd
            h    = invphi * h
            d_pt = a + invphi * h
            yd   = -_ev_single(d_pt, s, d_col, c_col, p, vinterp_vals, s_lo, inv_h, fc, c_param, delta, beta, quad_weights)

    if yc < yd:
        n_opt   = c_pt
        val_max = -yc
    else:
        n_opt   = d_pt
        val_max = -yd

    no_order = _ev_single(0.0, s, d_col, c_col, p, vinterp_vals, s_lo, inv_h, fc, c_param, delta, beta, quad_weights)
    if val_max < no_order:
        return 0.0, no_order
    return n_opt, val_max


def solve_price_policy(params, c_tilde, omega):
    p_policy = np.empty(params.ns, dtype=np.float64)
    price_converged = True
    p_unc = limit_price(params.mu_log_nu, params.sigma_log_nu, params.epsilon, params.gamma, c_tilde, omega)
    p_lo = 0.99 * p_unc
    p_hi = 50.0
    # Minimum nu_bar that keeps f_bar above the 1e-10 guard in price_residual.
    # For small s, p_lo may fall below this threshold, giving residual=1e6 on
    # both bracket ends (no sign change). We lift the lower bound per-state.
    _nu_threshold = math.exp(params.mu_log_nu + params.sigma_log_nu * float(ndtri(1e-8)))
    for i, s in enumerate(params.sgrid):
        if s == 0.0:
            p_policy[i] = 1.0
            continue
        p_lo_s = max(p_lo, (_nu_threshold / s) ** (1.0 / params.epsilon))
        try:
            p_opt = brentq(
                lambda p: price_residual(p, s, c_tilde, omega, params),
                p_lo_s, p_hi,
                xtol=1e-12,
            )
            p_policy[i] = p_opt
            if proxy_profit(p_opt, s, c_tilde, omega, params) < 0.0:
                price_converged = False
        except ValueError:
            price_converged = False
            p_policy[i] = p_unc
    return p_policy, price_converged


def precompute_demand(p_policy, params):
    # Build (ns, n_omega, Q) then transpose to (n_omega, ns, Q) so the prange-j
    # outer loop in _vfi_sweep accesses contiguous (ns, Q) slabs per thread.
    p_neg_eps = p_policy ** (-params.epsilon)
    d_table = np.minimum(
        params.quad_nodes_lognormal[None, None, :] * p_neg_eps[:, :, None],
        params.sgrid[:, None, None],
    )
    c_table = params.omega_grid[None, :, None] * (d_table ** params.gamma)
    # transpose(1, 0, 2): (ns, n_omega, Q) -> (n_omega, ns, Q)
    return np.ascontiguousarray(d_table.transpose(1, 0, 2)), np.ascontiguousarray(c_table.transpose(1, 0, 2))


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _vfi_sweep(ev_all, d_table, c_table, p_policy, sgrid,
               s_lo, inv_h, fc, c_param, delta, beta, quad_weights):
    """Single compiled VFI sweep over all inventory and omega states.
    ev_all: (n_omega, ns) — row j is the continuation-value vector for omega j.
    d_table, c_table: (n_omega, ns, Q) — j-axis outer; each thread owns one contiguous (ns, Q) slab.
    Parallelized over all n_omega * ns cells; each cell is independent (n_upper fixed at sgrid[-1]).
    """
    n_omega = d_table.shape[0]
    ns      = d_table.shape[1]
    n_upper = sgrid[ns - 1]
    v_new_t = np.zeros((n_omega, ns))
    n_pol_t = np.zeros((n_omega, ns))
    for idx in prange(n_omega * ns):
        j     = idx // ns
        i     = idx  % ns
        s     = sgrid[i]
        d_col = d_table[j, i, :]
        c_col = c_table[j, i, :]
        p     = p_policy[i, j]
        ev_j  = ev_all[j, :]
        n_t, v_t = _optimize_n(
            s, d_col, c_col, p, ev_j, s_lo, inv_h,
            fc, c_param, delta, beta, quad_weights, n_upper,
        )
        v_new_t[j, i] = v_t
        n_pol_t[j, i] = n_t
    return v_new_t.T, n_pol_t.T


def _maximize_expected_value_choice_precomp(s_i, j, d_table, c_table, p_policy, vinterp, params, n_upper):
    # d_table shape is (n_omega, ns, Q) — last axis already contiguous, no copy needed
    d_col = d_table[j, s_i, :]
    c_col = c_table[j, s_i, :]
    p = p_policy[s_i, j]
    s = params.sgrid[s_i]
    return _optimize_n(
        s, d_col, c_col, p,
        vinterp.values, vinterp.s_lo, vinterp.inv_h,
        params.fc, params.c, params.delta, params.beta,
        params.quad_weights, n_upper,
    )


def solve_value_function(params, tol=1e-4, maxiter=1000,conv="policy"):
    sgrid   = params.sgrid
    ns      = params.ns
    n_omega = params.q_omega
    p_omega = params.p_omega

    v_by_omega       = np.zeros((ns, n_omega), dtype=np.float64)
    p_policy_current = np.zeros((ns, n_omega), dtype=np.float64)
    n_policy_prev = np.zeros((ns, n_omega), dtype=np.float64)
    all_price_converged = True
    for j in range(n_omega):
        p_col, p_conv = solve_price_policy(params, params.c, params.omega_grid[j])
        p_policy_current[:, j] = p_col
        if not p_conv:
            all_price_converged = False
    
    if not all_price_converged:
        return {
            "v":          v_by_omega @ params.pi_omega,
            "n_policy":   np.zeros((ns, n_omega), dtype=np.float64),
            "p_policy":   p_policy_current,
            "v_by_omega": v_by_omega,
            "converged":  False,
            "iterations": 0,
            "final_diff": float("inf"),
        }

    d_table, c_table = precompute_demand(p_policy_current, params)

    inv_h = (ns - 1) / (sgrid[-1] - sgrid[0])
    s_lo  = float(sgrid[0])

    diff = float("inf")
    it   = 0

    while diff > tol and it < maxiter:
        # One BLAS-3 matmul replaces 7 separate matrix-vector products.
        # ev_all[j, :] = E[V(s, omega') | omega_j]  —  shape (n_omega, ns), rows contiguous.
        ev_all = p_omega @ v_by_omega.T

        v_by_omega_new, n_policy_current = _vfi_sweep(
            ev_all, d_table, c_table, p_policy_current, sgrid,
            s_lo, inv_h,
            params.fc, params.c, params.delta, params.beta,
            params.quad_weights,
        )

        if conv == "policy" and it > 10:
            diff = float(np.mean(np.abs(n_policy_current - n_policy_prev)))
            n_policy_prev = n_policy_current.copy()
        else:
            diff = float(np.max(np.abs(v_by_omega_new - v_by_omega)))
        v_by_omega = v_by_omega_new
        it        += 1

    v         = v_by_omega @ params.pi_omega
    converged = diff <= tol

    return {
        "v":          v,
        "n_policy":   n_policy_current,
        "p_policy":   p_policy_current,
        "v_by_omega": v_by_omega,
        "converged":  converged,
        "iterations": it,
        "final_diff": diff,
    }
