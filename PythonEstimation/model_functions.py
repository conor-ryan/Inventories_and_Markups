from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


SQRT_PI = float(np.sqrt(np.pi))


def _tauchen_log_ar1(mu_log: float, rho: float, sigma_eta: float, q_omega: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sigma_eta <= 0.0 or q_omega <= 1:
        omega_grid = np.array([np.exp(mu_log)], dtype=float)
        p_omega = np.ones((1, 1), dtype=float)
        pi_omega = np.array([1.0], dtype=float)
        return omega_grid, p_omega, pi_omega

    sigma_log_omega = sigma_eta / np.sqrt(max(1.0 - rho * rho, 1e-12))
    m_tau = 3.0
    log_omega_grid = np.linspace(mu_log - m_tau * sigma_log_omega, mu_log + m_tau * sigma_log_omega, q_omega)
    h = log_omega_grid[1] - log_omega_grid[0]

    p_omega = np.zeros((q_omega, q_omega), dtype=float)
    for i in range(q_omega):
        cond_mean = mu_log + rho * (log_omega_grid[i] - mu_log)
        for j in range(q_omega):
            if j == 0:
                p_omega[i, j] = norm.cdf((log_omega_grid[j] + h / 2.0 - cond_mean) / sigma_eta)
            elif j == q_omega - 1:
                p_omega[i, j] = 1.0 - norm.cdf((log_omega_grid[j] - h / 2.0 - cond_mean) / sigma_eta)
            else:
                hi = norm.cdf((log_omega_grid[j] + h / 2.0 - cond_mean) / sigma_eta)
                lo = norm.cdf((log_omega_grid[j] - h / 2.0 - cond_mean) / sigma_eta)
                p_omega[i, j] = hi - lo

        row_sum = p_omega[i, :].sum()
        if row_sum > 0.0:
            p_omega[i, :] /= row_sum

    pi_omega = np.ones(q_omega, dtype=float) / q_omega
    for _ in range(2000):
        pi_omega = p_omega.T @ pi_omega
    pi_omega /= pi_omega.sum()
    return np.exp(log_omega_grid), p_omega, pi_omega


@dataclass
class Parameters:
    c: float = 1.0
    fc: float = 0.0
    mu_eta: float = 0.0
    sigma_eta2: float = 0.0
    rho_omega: float = 0.9
    gamma: float = 1.0
    delta: float = 0.2
    beta: float = 0.95
    epsilon: float = 2.0
    mu_nu: float = 1.0
    sigma_nu2: float = 0.15
    q: int = 19
    q_omega: int = 7
    scale: float = 1.0
    size: float = 100.0
    ns: int = 200

    quad_nodes: np.ndarray = field(init=False, repr=False)
    quad_weights: np.ndarray = field(init=False, repr=False)
    gl_nodes: np.ndarray = field(init=False, repr=False)
    gl_weights: np.ndarray = field(init=False, repr=False)
    quad_nodes_lognormal: np.ndarray = field(init=False, repr=False)

    omega_grid: np.ndarray = field(init=False, repr=False)
    p_omega: np.ndarray = field(init=False, repr=False)
    pi_omega: np.ndarray = field(init=False, repr=False)

    dist_mu: float = field(init=False)
    dist_sigma: float = field(init=False)
    smax: float = field(init=False)
    sgrid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x, w = np.polynomial.hermite.hermgauss(self.q)
        gl_x, gl_w = np.polynomial.legendre.leggauss(self.q)

        scale_parameter = self.scale ** self.epsilon

        c_scaled = self.c * self.scale
        mu_nu_scaled = self.mu_nu * scale_parameter * self.size
        sigma_nu2_scaled = self.sigma_nu2 * (scale_parameter ** 2) * (self.size ** 2)
        mu_eta_shifted = self.mu_eta + (1.0 - self.rho_omega) * np.log(self.scale) + (1.0 - self.rho_omega) * np.log(self.size ** (1.0 - self.gamma))

        sigma2 = np.log(1.0 + sigma_nu2_scaled / (mu_nu_scaled ** 2))
        sigma = np.sqrt(max(sigma2, 1e-14))
        mu = np.log(mu_nu_scaled) - 0.5 * sigma2

        quad_nodes_lognormal = np.exp(mu + np.sqrt(2.0) * sigma * x)

        mu_log_omega = mu_eta_shifted / (1.0 - self.rho_omega) if abs(1.0 - self.rho_omega) > 1e-10 else mu_eta_shifted
        sigma_eta = np.sqrt(max(mu_eta_shifted * 0.0 + self.sigma_eta2, 0.0))
        omega_grid, p_omega, pi_omega = _tauchen_log_ar1(mu_log_omega, self.rho_omega, sigma_eta, self.q_omega)

        # Keep level moments in original scale (matching Julia constructor's final reset).
        self.c = float(c_scaled)
        self.mu_eta = float(mu_eta_shifted)
        self.mu_nu = float(mu_nu_scaled / (scale_parameter * self.size))
        self.sigma_nu2 = float(sigma_nu2_scaled / ((scale_parameter ** 2) * (self.size ** 2)))

        self.quad_nodes = x.astype(float)
        self.quad_weights = w.astype(float)
        self.gl_nodes = gl_x.astype(float)
        self.gl_weights = gl_w.astype(float)
        self.quad_nodes_lognormal = quad_nodes_lognormal.astype(float)

        self.omega_grid = omega_grid.astype(float)
        self.p_omega = p_omega.astype(float)
        self.pi_omega = pi_omega.astype(float)
        self.q_omega = int(self.omega_grid.size)

        self.dist_mu = float(mu)
        self.dist_sigma = float(sigma)
        self.smax = float(np.exp(self.dist_mu + self.dist_sigma * norm.ppf(0.85)) * (self.epsilon - 1.0) / self.epsilon)
        self.sgrid = np.linspace(1e-4, self.smax, self.ns).astype(float)

    def clone_with(self, **kwargs: float) -> "Parameters":
        return replace(self, **kwargs)


class UniformInterp:
    def __init__(self, values: np.ndarray, s_lo: float, inv_h: float):
        self.values = np.asarray(values, dtype=float)
        self.s_lo = float(s_lo)
        self.inv_h = float(inv_h)

    def __call__(self, x: float) -> float:
        t = (x - self.s_lo) * self.inv_h
        n = self.values.size
        if t <= 0.0:
            return float(self.values[0] + t * (self.values[1] - self.values[0]))
        if t >= n - 1:
            excess = t - (n - 1)
            return float(self.values[-1] + excess * (self.values[-1] - self.values[-2]))
        i = int(np.floor(t))
        alpha = t - i
        return float(self.values[i] + alpha * (self.values[i + 1] - self.values[i]))


class OmegaPolicyInterp:
    def __init__(self, nodes: list[UniformInterp], omega_grid: np.ndarray):
        self.nodes = nodes
        self.omega_grid = np.asarray(omega_grid, dtype=float)

    def __call__(self, x: float, omega_idx_or_value: int | float) -> float:
        if isinstance(omega_idx_or_value, (int, np.integer)):
            return self.nodes[int(omega_idx_or_value)](x)
        j = int(np.argmin(np.abs(self.omega_grid - float(omega_idx_or_value))))
        return self.nodes[j](x)


def build_fast_policy_interpolants(sgrid: np.ndarray, p_policy: np.ndarray, order_policy: np.ndarray) -> tuple[list[UniformInterp], list[UniformInterp]]:
    inv_h = (len(sgrid) - 1) / (sgrid[-1] - sgrid[0])
    s_lo = float(sgrid[0])
    price_nodes = [UniformInterp(p_policy[:, j], s_lo, inv_h) for j in range(p_policy.shape[1])]
    order_nodes = [UniformInterp(order_policy[:, j], s_lo, inv_h) for j in range(order_policy.shape[1])]
    return price_nodes, order_nodes


def truncated_lognormal_mean(nubar: float, params: Parameters) -> float:
    z1 = (np.log(nubar) - params.dist_mu - params.dist_sigma ** 2) / params.dist_sigma
    z0 = (np.log(nubar) - params.dist_mu) / params.dist_sigma
    denom = norm.cdf(z0)
    if denom <= 1e-14:
        return 0.0
    return float(np.exp(params.dist_mu + 0.5 * params.dist_sigma ** 2) * norm.cdf(z1) / denom)


def truncated_lognormal_ratio_e_nu_gamma_e_nu(nubar: float, params: Parameters) -> tuple[float, float, float]:
    fbar = norm.cdf((np.log(nubar) - params.dist_mu) / params.dist_sigma)
    if fbar < 1e-10:
        return 0.0, 0.0, 0.0

    u_q = 0.5 * fbar * (params.gl_nodes + 1.0)
    nu_q = np.exp(params.dist_mu + params.dist_sigma * norm.ppf(np.clip(u_q, 1e-14, 1.0 - 1e-14)))
    num = float(np.sum(params.gl_weights * (nu_q ** params.gamma)))
    den = float(np.sum(params.gl_weights * nu_q))
    ratio = num / den if den > 0.0 else 0.0
    return ratio, 0.5 * num, 0.5 * den


def price_residual(p: float, s: float, c_tilde: float, omega: float, params: Parameters) -> float:
    nubar = s * (p ** params.epsilon)
    z = (np.log(nubar) - params.dist_mu) / params.dist_sigma
    fbar = norm.cdf(z)
    tail = 1.0 - fbar
    if fbar < 1e-10:
        return 1e6

    e_nu = truncated_lognormal_mean(nubar, params)
    ratio_e_nu_gamma = truncated_lognormal_ratio_e_nu_gamma_e_nu(nubar, params)[0]

    opp_mc = omega * params.gamma * ratio_e_nu_gamma * (p ** (params.epsilon * (1.0 - params.gamma)))
    rhs = (params.epsilon / (params.epsilon - 1.0)) * (opp_mc + c_tilde)
    rhs += (1.0 / (params.epsilon - 1.0)) * s * (p ** (params.epsilon + 1.0)) * (1.0 / max(e_nu, 1e-14)) * (tail / max(fbar, 1e-14))
    return float(p - rhs)


def solve_price_policy(params: Parameters, c_tilde: float, omega: float) -> np.ndarray:
    p_policy = np.zeros(params.ns, dtype=float)
    for i, s in enumerate(params.sgrid):
        obj = lambda p: price_residual(float(p), float(s), c_tilde, omega, params) ** 2
        res = minimize_scalar(obj, method="bounded", bounds=(1e-3, 50.0), options={"xatol": 1e-10, "maxiter": 500})
        p_policy[i] = float(res.x)
    return p_policy


def precompute_demand(p_policy: np.ndarray, params: Parameters) -> tuple[np.ndarray, np.ndarray]:
    q, n_omega, ns = params.q, params.q_omega, params.ns
    d_table = np.empty((q, n_omega, ns), dtype=float)
    c_table = np.empty((q, n_omega, ns), dtype=float)

    for i in range(ns):
        s = params.sgrid[i]
        for j in range(n_omega):
            p = p_policy[i, j]
            p_neg_eps = p ** (-params.epsilon)
            omega = params.omega_grid[j]
            d = np.minimum(params.quad_nodes_lognormal * p_neg_eps, s)
            d_table[:, j, i] = d
            c_table[:, j, i] = omega * (d ** params.gamma)

    return d_table, c_table


def _shock_specific_value_precomp(n: float, d: float, c: float, p: float, s_i: int, vinterp: UniformInterp, params: Parameters) -> float:
    s = params.sgrid[s_i]
    s_tilde = s - d + n
    order_cost = params.fc + params.c * n if n > 0.0 else 0.0
    return float(p * d - c - order_cost + params.beta * vinterp((1.0 - params.delta) * s_tilde))


def _uniform_interp_eval_array(values: np.ndarray, s_lo: float, inv_h: float, x: np.ndarray) -> np.ndarray:
    t = (x - s_lo) * inv_h
    n = values.size
    out = np.empty_like(t, dtype=float)

    left = t <= 0.0
    right = t >= (n - 1)
    mid = ~(left | right)

    if np.any(left):
        out[left] = values[0] + t[left] * (values[1] - values[0])
    if np.any(right):
        excess = t[right] - (n - 1)
        out[right] = values[-1] + excess * (values[-1] - values[-2])
    if np.any(mid):
        t_mid = t[mid]
        i = np.floor(t_mid).astype(np.int64)
        alpha = t_mid - i
        out[mid] = values[i] + alpha * (values[i + 1] - values[i])

    return out


def _expected_value_choice_precomp_fast(
    n: float,
    s_val: float,
    d_col: np.ndarray,
    base_profit_col: np.ndarray,
    quad_weights: np.ndarray,
    beta: float,
    delta: float,
    fixed_cost: float,
    unit_cost: float,
    vinterp_values: np.ndarray,
    vinterp_s_lo: float,
    vinterp_inv_h: float,
) -> float:
    s_tilde = s_val - d_col + n
    x_next = (1.0 - delta) * s_tilde
    v_next = _uniform_interp_eval_array(vinterp_values, vinterp_s_lo, vinterp_inv_h, x_next)
    order_cost = fixed_cost + unit_cost * n if n > 0.0 else 0.0
    payoffs = base_profit_col - order_cost + beta * v_next
    return float(np.dot(quad_weights, payoffs) / SQRT_PI)


def _maximize_expected_value_choice_precomp(
    s_i: int,
    j: int,
    d_table: np.ndarray,
    c_table: np.ndarray,
    p_policy: np.ndarray,
    vinterp: UniformInterp,
    params: Parameters,
    n_upper: float,
) -> tuple[float, float]:
    d_col = d_table[:, j, s_i]
    c_col = c_table[:, j, s_i]
    p = p_policy[s_i, j]
    s_val = float(params.sgrid[s_i])
    base_profit_col = p * d_col - c_col

    vw = params.quad_weights
    beta = params.beta
    delta = params.delta
    fixed_cost = params.fc
    unit_cost = params.c
    vinterp_values = vinterp.values
    vinterp_s_lo = vinterp.s_lo
    vinterp_inv_h = vinterp.inv_h

    def obj(n: float) -> float:
        return -_expected_value_choice_precomp_fast(
            float(n),
            s_val,
            d_col,
            base_profit_col,
            vw,
            beta,
            delta,
            fixed_cost,
            unit_cost,
            vinterp_values,
            vinterp_s_lo,
            vinterp_inv_h,
        )

    res = minimize_scalar(obj, method="bounded", bounds=(0.0, float(n_upper)), options={"xatol": 1e-6, "maxiter": 200})

    n_opt = float(res.x)
    value_max = float(-res.fun)
    no_order = _expected_value_choice_precomp_fast(
        0.0,
        s_val,
        d_col,
        base_profit_col,
        vw,
        beta,
        delta,
        fixed_cost,
        unit_cost,
        vinterp_values,
        vinterp_s_lo,
        vinterp_inv_h,
    )

    if value_max < no_order:
        return 0.0, no_order
    return n_opt, value_max


def solve_value_function(
    params: Parameters,
    tol: float = 1e-4,
    maxiter: int = 1000,
    full: bool = False,
    fast_interp: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    if full:
        raise NotImplementedError("full=True path is not implemented in the Python translation yet.")

    ns, n_omega = params.ns, params.q_omega
    v_by_omega = np.zeros((ns, n_omega), dtype=float)
    v_by_omega_new = np.zeros_like(v_by_omega)

    n_policy = np.zeros((ns, n_omega), dtype=float)
    p_policy = np.zeros((ns, n_omega), dtype=float)

    for j in range(n_omega):
        p_policy[:, j] = solve_price_policy(params, params.c, float(params.omega_grid[j]))

    d_table, c_table = precompute_demand(p_policy, params)

    inv_h = (ns - 1) / (params.sgrid[-1] - params.sgrid[0])
    diff = np.inf
    it = 0
    t0 = time.perf_counter()

    while diff > tol and it < maxiter:
        for j in range(n_omega):
            ev_cont_j = v_by_omega @ params.p_omega[j, :]
            if fast_interp:
                vinterp_j = UniformInterp(ev_cont_j.copy(), float(params.sgrid[0]), float(inv_h))
            else:
                vinterp_j = UniformInterp(ev_cont_j.copy(), float(params.sgrid[0]), float(inv_h))

            n_upper = float(params.sgrid[-1])
            for i in range(ns):
                n_t, v_t = _maximize_expected_value_choice_precomp(i, j, d_table, c_table, p_policy, vinterp_j, params, n_upper=n_upper)
                v_by_omega_new[i, j] = v_t
                n_policy[i, j] = n_t
                if n_t > 0.0:
                    n_upper = n_t

        diff = float(np.max(np.abs(v_by_omega_new - v_by_omega)))
        v_by_omega[:, :] = v_by_omega_new
        it += 1
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"VFI iter={it:4d} diff={diff:12.6e} elapsed={elapsed:9.2f}s", flush=True)

    if verbose:
        total = time.perf_counter() - t0
        print(f"Initial value function solved in {it} iterations ({total:.2f}s)", flush=True)

    v = v_by_omega @ params.pi_omega
    converged = bool(diff <= tol)
    return v, n_policy, p_policy, v_by_omega, converged


def solve_model(
    params: Parameters,
    full: bool = False,
    verbose: bool = False,
    fast_interp: bool = True,
    maxiter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, OmegaPolicyInterp, OmegaPolicyInterp, UniformInterp, bool]:
    if verbose:
        print("Solving value function...")

    v, order_policy, p_policy, v_by_omega, converged = solve_value_function(
        params,
        full=full,
        fast_interp=fast_interp,
        maxiter=maxiter,
        verbose=verbose,
    )

    price_nodes, order_nodes = build_fast_policy_interpolants(params.sgrid, p_policy, order_policy)
    price_policy_interp = OmegaPolicyInterp(price_nodes, params.omega_grid)
    order_policy_interp = OmegaPolicyInterp(order_nodes, params.omega_grid)
    inv_h = (params.ns - 1) / (params.sgrid[-1] - params.sgrid[0])
    vinterp = UniformInterp(v, float(params.sgrid[0]), float(inv_h))

    return p_policy, order_policy, v, v_by_omega, price_policy_interp, order_policy_interp, vinterp, converged


def operating_expense(omega: float, demand: float, params: Parameters) -> float:
    return float(omega * (demand ** params.gamma))


def draw_omega_index(rng: np.random.Generator, params: Parameters, current_idx: int) -> int:
    probs = params.p_omega[current_idx, :]
    return int(rng.choice(params.q_omega, p=probs))


def draw_omega_index_ergodic(rng: np.random.Generator, params: Parameters) -> int:
    return int(rng.choice(params.q_omega, p=params.pi_omega))


def simulate_firm(
    num_simulations: int,
    num_periods: int,
    price_policy_interp: OmegaPolicyInterp,
    order_policy_interp: OmegaPolicyInterp,
    params: Parameters,
    burn_in: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng

    inventory = np.empty(num_simulations * num_periods, dtype=float)
    demand = np.empty_like(inventory)
    expenses = np.empty_like(inventory)
    revenue = np.empty_like(inventory)

    write_idx = 0
    for _sim in range(num_simulations):
        s_current = float(rng.choice(params.sgrid))
        omega_idx = draw_omega_index_ergodic(rng, params)

        for _ in range(burn_in):
            p_opt = price_policy_interp(s_current, omega_idx)
            n_opt = order_policy_interp(s_current, omega_idx)
            nu = float(np.exp(params.dist_mu + params.dist_sigma * rng.standard_normal()))
            d = min(nu * (p_opt ** (-params.epsilon)), s_current)
            s_current = max((1.0 - params.delta) * (s_current - d + n_opt), 0.0)
            omega_idx = draw_omega_index(rng, params, omega_idx)

        for _ in range(num_periods):
            inventory[write_idx] = s_current
            p_opt = price_policy_interp(s_current, omega_idx)
            n_opt = order_policy_interp(s_current, omega_idx)
            omega_current = float(params.omega_grid[omega_idx])

            nu = float(np.exp(params.dist_mu + params.dist_sigma * rng.standard_normal()))
            d = min(nu * (p_opt ** (-params.epsilon)), s_current)
            demand[write_idx] = d
            expenses[write_idx] = operating_expense(omega_current, d, params)
            revenue[write_idx] = p_opt * d

            s_end = s_current - d
            s_current = (1.0 - params.delta) * (s_end + n_opt)
            omega_idx = draw_omega_index(rng, params, omega_idx)
            write_idx += 1

    return inventory, demand, expenses, revenue
