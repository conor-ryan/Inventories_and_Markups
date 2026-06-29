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


# ──────────────────────────────────────────────────────────────────────────────
# JIT-compatible normal distribution primitives
# ──────────────────────────────────────────────────────────────────────────────

@jit(nopython=True, fastmath=True, cache=True)
def _norm_cdf(x):
    return 0.5 * math.erfc(-x * 0.7071067811865476)


@jit(nopython=True, fastmath=True, cache=True)
def _norm_ppf(p):
    # Peter Acklam's rational approximation; max absolute error ~1.15e-9
    a0 = -3.969683028665376e+01; a1 =  2.209460984245205e+02
    a2 = -2.759285104469687e+02; a3 =  1.383577518672690e+02
    a4 = -3.066479806614716e+01; a5 =  2.506628277459239e+00
    b0 = -5.447609879822406e+01; b1 =  1.615858368580409e+02
    b2 = -1.556989798598866e+02; b3 =  6.680131188771972e+01
    b4 = -1.328068155288572e+01
    c0 = -7.784894002430293e-03; c1 = -3.223964580411365e-01
    c2 = -2.400758277161838e+00; c3 = -2.549732539343734e+00
    c4 =  4.374664141464968e+00; c5 =  2.938163982698783e+00
    d0 =  7.784695709041462e-03; d1 =  3.224671290700398e-01
    d2 =  2.445134137142996e+00; d3 =  3.754408661907416e+00
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / \
               ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)
    elif p <= 1.0 - p_low:
        q = p - 0.5; r = q * q
        return (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)*q / \
               (((((b0*r+b1)*r+b2)*r+b3)*r+b4)*r+1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / \
                ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)


# ──────────────────────────────────────────────────────────────────────────────
# JIT versions of price helper functions (raw scalars/arrays, not params object)
# ──────────────────────────────────────────────────────────────────────────────

@jit(nopython=True, fastmath=True, cache=True)
def _lognormal_cdf_jit(x, mu, sigma):
    if x <= 0.0:
        return 0.0
    return _norm_cdf((math.log(x) - mu) / sigma)


@jit(nopython=True, fastmath=True, cache=True)
def _trunc_lognormal_mean_jit(nu_bar, mu_log_nu, sigma_log_nu):
    log_nu = math.log(nu_bar)
    z1 = (log_nu - mu_log_nu - sigma_log_nu * sigma_log_nu) / sigma_log_nu
    z0 = (log_nu - mu_log_nu) / sigma_log_nu
    return math.exp(mu_log_nu + 0.5 * sigma_log_nu * sigma_log_nu) * _norm_cdf(z1) / _norm_cdf(z0)


@jit(nopython=True, fastmath=True, cache=True)
def _trunc_ratio_jit(nu_bar, mu_log_nu, sigma_log_nu, gamma, gl_nodes, gl_weights):
    """(ratio, half_num, half_den) = (E[ν^γ]/E[ν], ½·f_bar·E[ν^γ], ½·f_bar·E[ν]) under truncated lognormal."""
    f_bar = _lognormal_cdf_jit(nu_bar, mu_log_nu, sigma_log_nu)
    if f_bar <= 0.0:
        return 0.0, 0.0, 0.0
    half_fbar = 0.5 * f_bar
    num = 0.0; den = 0.0
    for k in range(len(gl_nodes)):
        u_k = half_fbar * (gl_nodes[k] + 1.0)
        if u_k < 1e-12:
            u_k = 1e-12
        elif u_k > 1.0 - 1e-12:
            u_k = 1.0 - 1e-12
        nu_k = math.exp(mu_log_nu + sigma_log_nu * _norm_ppf(u_k))
        w_k  = gl_weights[k]
        num += w_k * nu_k ** gamma
        den += w_k * nu_k
    ratio = (num / den) if den > 0.0 else 0.0
    return ratio, 0.5 * num, 0.5 * den


@jit(nopython=True, fastmath=True, cache=True)
def _proxy_profit_unconst_jit(p, c_tilde, omega, mu_log_nu, sigma_log_nu, epsilon, gamma):
    e_nu       = math.exp(mu_log_nu + 0.5 * sigma_log_nu * sigma_log_nu)
    e_nu_gamma = math.exp(gamma * mu_log_nu + 0.5 * (gamma * sigma_log_nu) ** 2)
    return e_nu * p ** (-epsilon) * (p - c_tilde) - omega * e_nu_gamma * p ** (-gamma * epsilon)


@jit(nopython=True, fastmath=True, cache=True)
def _price_residual_jit(p, s, c_tilde, omega, epsilon, gamma,
                        mu_log_nu, sigma_log_nu, gl_nodes, gl_weights):
    nu_bar = s * (p ** epsilon)
    f_bar  = _lognormal_cdf_jit(nu_bar, mu_log_nu, sigma_log_nu)
    if f_bar < 1e-10:
        return 1e6
    tail  = 1.0 - f_bar
    e_nu  = _trunc_lognormal_mean_jit(nu_bar, mu_log_nu, sigma_log_nu)
    ratio = _trunc_ratio_jit(nu_bar, mu_log_nu, sigma_log_nu, gamma, gl_nodes, gl_weights)[0]
    opp_mc = omega * gamma * ratio * (p ** (epsilon * (1.0 - gamma)))
    rhs = (
        (epsilon / (epsilon - 1.0)) * (opp_mc + c_tilde)
        + (1.0 / (epsilon - 1.0)) * s * (p ** (epsilon + 1.0)) * (1.0 / e_nu) * (tail / f_bar)
    )
    return p - rhs


@jit(nopython=True, fastmath=True, cache=True)
def _proxy_profit_jit(p, s, c_tilde, omega, epsilon, gamma,
                      mu_log_nu, sigma_log_nu, gl_nodes, gl_weights):
    nu_bar = s * p ** epsilon
    f_bar  = _lognormal_cdf_jit(nu_bar, mu_log_nu, sigma_log_nu)
    tail   = 1.0 - f_bar
    _, e_nu_gamma_trunc, e_nu_trunc = _trunc_ratio_jit(
        nu_bar, mu_log_nu, sigma_log_nu, gamma, gl_nodes, gl_weights
    )
    e_d       = p ** (-epsilon) * f_bar * e_nu_trunc + s * tail
    e_d_gamma = p ** (-gamma * epsilon) * f_bar * e_nu_gamma_trunc + (s ** gamma) * tail
    return (p - c_tilde) * e_d - omega * e_d_gamma


# ──────────────────────────────────────────────────────────────────────────────
# JIT root/minimization solvers — Brent's method, inlined objective
# ──────────────────────────────────────────────────────────────────────────────

@jit(nopython=True, fastmath=True, cache=True)
def _limit_price_jit(c_tilde, omega, mu_log_nu, sigma_log_nu, epsilon, gamma):
    """Brent's bounded minimization of -proxy_profit_unconst on [1e-3, 5.0]."""
    CGOLD = 0.3819660112501051
    ZEPS  = 1e-10
    tol   = 1e-8
    a = 1e-3; b = 5.0

    x = w = v = a + CGOLD * (b - a)
    fx = fw = fv = -_proxy_profit_unconst_jit(x, c_tilde, omega, mu_log_nu, sigma_log_nu, epsilon, gamma)
    d = e = 0.0

    for _ in range(500):
        xm   = 0.5 * (a + b)
        tol1 = tol * abs(x) + ZEPS
        tol2 = 2.0 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            return x
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p_val = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p_val = -p_val
            else:
                q = -q
            r = e; e = d
            if abs(p_val) < abs(0.5 * q * r) and p_val > q * (a - x) and p_val < q * (b - x):
                d = p_val / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if xm >= x else -tol1
            else:
                e = (a - x) if x >= xm else (b - x)
                d = CGOLD * e
        else:
            e = (a - x) if x >= xm else (b - x)
            d = CGOLD * e
        u  = x + (d if abs(d) >= tol1 else (tol1 if d >= 0.0 else -tol1))
        fu = -_proxy_profit_unconst_jit(u, c_tilde, omega, mu_log_nu, sigma_log_nu, epsilon, gamma)
        if fu <= fx:
            if u < x: b = x
            else:     a = x
            v = w; fv = fw; w = x; fw = fx; x = u; fx = fu
        else:
            if u < x: a = u
            else:     b = u
            if fu <= fw or w == x:
                v = w; fv = fw; w = u; fw = fu
            elif fu <= fv or v == x or v == w:
                v = u; fv = fu
    return x


@jit(nopython=True, fastmath=True, cache=True)
def _maximize_proxy_profit_jit(s, c_tilde, omega, p_lo, p_hi, epsilon, gamma,
                                mu_log_nu, sigma_log_nu, gl_nodes, gl_weights):
    """Brent's bounded maximization of proxy_profit on [p_lo, p_hi].

    Fallback for _brent_root_price when no sign change exists in the bracket.
    Returns (p_opt, profit_at_p_opt). Caller must verify profit_at_p_opt >= 0
    to confirm a valid (profitable) price was found rather than a boundary result.
    """
    CGOLD = 0.3819660112501051
    ZEPS  = 1e-10
    tol   = 1e-8
    a = p_lo; b = p_hi

    x = w = v = a + CGOLD * (b - a)
    fx = fw = fv = -_proxy_profit_jit(x, s, c_tilde, omega, epsilon, gamma,
                                       mu_log_nu, sigma_log_nu, gl_nodes, gl_weights)
    d = e = 0.0

    for _ in range(500):
        xm   = 0.5 * (a + b)
        tol1 = tol * abs(x) + ZEPS
        tol2 = 2.0 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            break
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p_val = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p_val = -p_val
            else:
                q = -q
            r = e; e = d
            if abs(p_val) < abs(0.5 * q * r) and p_val > q * (a - x) and p_val < q * (b - x):
                d = p_val / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if xm >= x else -tol1
            else:
                e = (a - x) if x >= xm else (b - x)
                d = CGOLD * e
        else:
            e = (a - x) if x >= xm else (b - x)
            d = CGOLD * e
        u  = x + (d if abs(d) >= tol1 else (tol1 if d >= 0.0 else -tol1))
        fu = -_proxy_profit_jit(u, s, c_tilde, omega, epsilon, gamma,
                                 mu_log_nu, sigma_log_nu, gl_nodes, gl_weights)
        if fu <= fx:
            if u < x: b = x
            else:     a = x
            v = w; fv = fw; w = x; fw = fx; x = u; fx = fu
        else:
            if u < x: a = u
            else:     b = u
            if fu <= fw or w == x:
                v = w; fv = fw; w = u; fw = fu
            elif fu <= fv or v == x or v == w:
                v = u; fv = fu

    profit = _proxy_profit_jit(x, s, c_tilde, omega, epsilon, gamma,
                                mu_log_nu, sigma_log_nu, gl_nodes, gl_weights)
    return x, profit


@jit(nopython=True, fastmath=True, cache=True)
def _brent_root_price(s, c_tilde, omega, p_lo, p_hi, epsilon, gamma,
                      mu_log_nu, sigma_log_nu, gl_nodes, gl_weights):
    """Brent's root finding for price_residual on [p_lo, p_hi]. Returns (p_opt, converged)."""
    xtol    = 1e-12
    EPS     = 2.220446049250313e-16
    maxiter = 200

    a = p_lo; b = p_hi
    fa = _price_residual_jit(a, s, c_tilde, omega, epsilon, gamma,
                             mu_log_nu, sigma_log_nu, gl_nodes, gl_weights)
    fb = _price_residual_jit(b, s, c_tilde, omega, epsilon, gamma,
                             mu_log_nu, sigma_log_nu, gl_nodes, gl_weights)

    if fa * fb > 0.0:
        return 0.5 * (a + b), False
    if fa == 0.0:
        return a, True
    if fb == 0.0:
        return b, True

    c = a; fc = fa; d = b - a; e = d

    for _ in range(maxiter):
        if fb * fc > 0.0:
            c = a; fc = fa; d = b - a; e = d
        if abs(fc) < abs(fb):
            a = b; fa = fb; b = c; fb = fc; c = a; fc = fa

        tol1 = 2.0 * EPS * abs(b) + 0.5 * xtol
        xm   = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b, True

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            ss = fb / fa
            if a == c:
                pp = 2.0 * xm * ss
                qq = 1.0 - ss
            else:
                qq = fa / fc; rr = fb / fc
                pp = ss * (2.0 * xm * qq * (qq - rr) - (b - a) * (rr - 1.0))
                qq = (qq - 1.0) * (rr - 1.0) * (ss - 1.0)
            if pp > 0.0:
                qq = -qq
            else:
                pp = -pp
            if 2.0 * pp < min(3.0 * xm * qq - abs(tol1 * qq), abs(e * qq)):
                e = d; d = pp / qq
            else:
                d = xm; e = d
        else:
            d = xm; e = d

        a = b; fa = fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0.0 else -tol1
        fb = _price_residual_jit(b, s, c_tilde, omega, epsilon, gamma,
                                 mu_log_nu, sigma_log_nu, gl_nodes, gl_weights)

    return b, abs(fb) <= xtol


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
        r_lo = price_residual(p_lo_s, s, c_tilde, omega, params)
        r_hi = price_residual(p_hi, s, c_tilde, omega, params)

        p_opt = None
        if r_lo * r_hi <= 0.0:
            try:
                p_opt = brentq(
                    lambda p: price_residual(p, s, c_tilde, omega, params),
                    p_lo_s, p_hi,
                    xtol=1e-12,
                )
                if proxy_profit(p_opt, s, c_tilde, omega, params) < 0.0:
                    p_opt = None
            except ValueError:
                p_opt = None

        if p_opt is None:
            result = minimize_scalar(
                lambda p: -proxy_profit(p, s, c_tilde, omega, params),
                bounds=(p_lo_s, p_hi),
                method='bounded',
            )
            p_opt_fb = result.x
            if proxy_profit(p_opt_fb, s, c_tilde, omega, params) >= 0.0:
                p_opt = p_opt_fb
            else:
                price_converged = False
                p_opt = p_unc

        p_policy[i] = p_opt
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


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _price_policy_sweep(sgrid, omega_grid, c_tilde, epsilon, gamma,
                        mu_log_nu, sigma_log_nu, gl_nodes, gl_weights):
    """Compute price policy for all (omega, s) cells in parallel.
    Returns (p_policy, all_converged) where p_policy has shape (ns, n_omega).
    """
    n_omega = len(omega_grid)
    ns      = len(sgrid)
    p_policy = np.zeros((ns, n_omega), dtype=np.float64)
    failed   = np.zeros(n_omega * ns, dtype=np.uint8)

    # Sequential: unconstrained prices per omega (7 iterations, cheap)
    p_unc = np.empty(n_omega, dtype=np.float64)
    for j in range(n_omega):
        p_unc[j] = _limit_price_jit(c_tilde, omega_grid[j], mu_log_nu, sigma_log_nu, epsilon, gamma)
    nu_threshold = math.exp(mu_log_nu + sigma_log_nu * _norm_ppf(1e-8))

    # Parallel: one Brent root-finding call per (omega, s) cell
    for idx in prange(n_omega * ns):
        j = idx // ns
        i = idx  % ns
        s = sgrid[i]
        if s == 0.0:
            p_policy[i, j] = 1.0
        else:
            p_unc_j = p_unc[j]
            p_lo_s  = max(0.99 * p_unc_j, (nu_threshold / s) ** (1.0 / epsilon))
            p_opt, conv = _brent_root_price(
                s, c_tilde, omega_grid[j], p_lo_s, 50.0,
                epsilon, gamma, mu_log_nu, sigma_log_nu, gl_nodes, gl_weights,
            )
            is_fail = not conv
            if not is_fail:
                if _proxy_profit_jit(
                    p_opt, s, c_tilde, omega_grid[j], epsilon, gamma,
                    mu_log_nu, sigma_log_nu, gl_nodes, gl_weights,
                ) < 0.0:
                    is_fail = True
            if is_fail:
                p_fb, profit_fb = _maximize_proxy_profit_jit(
                    s, c_tilde, omega_grid[j], p_lo_s, 50.0,
                    epsilon, gamma, mu_log_nu, sigma_log_nu, gl_nodes, gl_weights,
                )
                if profit_fb >= 0.0:
                    p_policy[i, j] = p_fb
                else:
                    failed[idx] = 1
                    p_policy[i, j] = p_unc_j
            else:
                p_policy[i, j] = p_opt

    return p_policy, np.sum(failed) == 0


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
    p_policy_current, all_price_converged = _price_policy_sweep(
        params.sgrid, params.omega_grid, params.c, params.epsilon, params.gamma,
        params.mu_log_nu, params.sigma_log_nu, params.gl_nodes, params.gl_weights,
    )
    
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
