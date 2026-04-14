from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model_functions import Parameters, simulate_firm, solve_model


MOMENT_NAMES = [
    "avg_isr",
    "var_log1p_isr",
    "avg_gross_margin",
    "γ_OLS",
    "ρ_ω",
    "σ_η2",
    "μ_η",
]


def _col(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    raise KeyError(f"Missing expected columns: {names}")


def estimate_omega_ar1(log_omega_proxy: np.ndarray, firm_boundary: np.ndarray) -> tuple[float, float, float, float, float, float]:
    n = len(log_omega_proxy)
    lag = np.full(n, np.nan, dtype=float)
    lag[1:] = log_omega_proxy[:-1]
    lag[firm_boundary] = np.nan

    keep = ~np.isnan(lag)
    y = log_omega_proxy[keep]
    x = lag[keep]
    t = len(y)

    x_bar, y_bar = float(x.mean()), float(y.mean())
    sxx = float(np.sum((x - x_bar) ** 2))
    rho = float(np.sum((x - x_bar) * (y - y_bar)) / sxx)
    a = y_bar - rho * x_bar

    resid = y - (a + rho * x)
    sigma_u2 = float(np.sum(resid ** 2) / max(t - 2, 1))

    se_rho = float(np.sqrt(sigma_u2 / sxx))
    se_a = float(np.sqrt(sigma_u2 * (1.0 / t + x_bar * x_bar / sxx)))

    mu_eta = a
    sigma_eta2 = sigma_u2
    se_mu_eta = se_a
    se_sigma_eta2 = float(np.sqrt(2.0 * sigma_eta2 * sigma_eta2 / max(t - 2, 1)))

    return mu_eta, sigma_eta2, rho, se_mu_eta, se_sigma_eta2, se_rho


def compute_annual_auxiliary(df_annual: pd.DataFrame) -> dict[str, Any]:
    df = df_annual.sort_values(["firm_id", "year_id"]).reset_index(drop=True)
    n = len(df)

    firm_boundary = np.zeros(n, dtype=bool)
    firm_boundary[0] = True
    firm_boundary[1:] = df["firm_id"].to_numpy()[1:] != df["firm_id"].to_numpy()[:-1]

    valid = (df["total_opex"] > 0) & (df["total_sales"] > 0)
    log_opex = np.log(df.loc[valid, "total_opex"].to_numpy(dtype=float))
    log_sales = np.log(df.loc[valid, "total_sales"].to_numpy(dtype=float))
    fb_valid = firm_boundary[valid.to_numpy()]

    x = log_sales
    y = log_opex
    x_bar = float(x.mean())
    y_bar = float(y.mean())
    sxx = float(np.sum((x - x_bar) ** 2))
    gamma_ols = float(np.sum((x - x_bar) * (y - y_bar)) / sxx)
    intercept = y_bar - gamma_ols * x_bar

    residuals = y - (intercept + gamma_ols * x)
    log_omega_proxy = intercept + residuals

    mu_eta, sigma_eta2, rho_omega, se_mu_eta, se_sigma_eta2, se_rho = estimate_omega_ar1(log_omega_proxy, fb_valid)

    return {
        "γ_OLS": gamma_ols,
        "ρ_ω": rho_omega,
        "σ_η2": sigma_eta2,
        "μ_η": mu_eta,
        "se_ρω": se_rho,
        "se_ση2": se_sigma_eta2,
        "se_μη": se_mu_eta,
    }


def compute_monthly_moments(df_monthly: pd.DataFrame) -> dict[str, float]:
    valid = df_monthly["revenue"] > 0
    isr = df_monthly.loc[valid, "inv_to_sales"].to_numpy(dtype=float)
    gm = (df_monthly.loc[valid, "revenue"] / df_monthly.loc[valid, "cogs"]).to_numpy(dtype=float)
    log1p_isr = np.log1p(isr)
    return {
        "avg_isr": float(isr.mean()),
        "var_log1p_isr": float(np.var(log1p_isr, ddof=1)),
        "avg_gross_margin": float(gm.mean()),
    }


def _simulate_all_moments(
    params: Parameters,
    ppi,
    opi,
    n_firms: int,
    n_years: int,
    seed: int | None,
) -> dict[str, float | bool]:
    n_months = n_years * 12
    rng = np.random.default_rng(seed)

    inv_sim, dem_sim, exp_sim, rev_sim = simulate_firm(
        num_simulations=n_firms,
        num_periods=n_months,
        price_policy_interp=ppi,
        order_policy_interp=opi,
        params=params,
        rng=rng,
    )
    any_inventory_above_grid = bool(np.any(inv_sim > params.smax))

    valid_mo = (dem_sim > 0) & (rev_sim > 0)
    isr_mo = inv_sim[valid_mo] / rev_sim[valid_mo]
    gm_mo = rev_sim[valid_mo] / (params.c * dem_sim[valid_mo])

    avg_isr_sim = float(isr_mo.mean())
    var_log1p_isr_sim = float(np.var(np.log1p(isr_mo), ddof=1))
    avg_gm_sim = float(gm_mo.mean())

    n_ann = n_firms * n_years
    firm_ids = np.empty(n_ann, dtype=int)
    year_ids = np.empty(n_ann, dtype=int)
    tot_opex = np.empty(n_ann, dtype=float)
    tot_sales = np.empty(n_ann, dtype=float)

    for firm in range(n_firms):
        m0 = firm * n_months
        a0 = firm * n_years
        for yr in range(n_years):
            m_first = m0 + yr * 12
            m_last = m0 + (yr + 1) * 12
            a_idx = a0 + yr
            firm_ids[a_idx] = firm + 1
            year_ids[a_idx] = yr + 1
            tot_opex[a_idx] = float(np.sum(exp_sim[m_first:m_last]))
            tot_sales[a_idx] = float(np.sum(dem_sim[m_first:m_last]))

    df_ann = pd.DataFrame(
        {
            "firm_id": firm_ids,
            "year_id": year_ids,
            "total_opex": tot_opex,
            "total_sales": tot_sales,
        }
    )
    psi_ann = compute_annual_auxiliary(df_ann)

    return {
        "avg_isr": avg_isr_sim,
        "var_log1p_isr": var_log1p_isr_sim,
        "avg_gross_margin": avg_gm_sim,
        "γ_OLS": float(psi_ann["γ_OLS"]),
        "ρ_ω": float(psi_ann["ρ_ω"]),
        "σ_η2": float(psi_ann["σ_η2"]),
        "μ_η": float(psi_ann["μ_η"]),
        "any_inventory_above_grid": any_inventory_above_grid,
    }


def _full_ii_target_vector(target_moments: dict[str, float]) -> np.ndarray:
    return np.array([target_moments[name] for name in MOMENT_NAMES], dtype=float)


def _full_ii_parameter_vector(params: Parameters) -> np.ndarray:
    return np.array([params.gamma, params.mu_eta, params.sigma_eta2, params.rho_omega, params.sigma_nu2, params.epsilon, params.delta], dtype=float)


def _full_ii_params_from_vector(params_base: Parameters, theta: np.ndarray) -> Parameters:
    return params_base.clone_with(
        gamma=float(theta[0]),
        mu_eta=float(theta[1]),
        sigma_eta2=float(theta[2]),
        rho_omega=float(theta[3]),
        sigma_nu2=float(theta[4]),
        epsilon=float(theta[5]),
        delta=float(theta[6]),
    )


def _full_ii_moment_vector_from_params(
    params: Parameters,
    n_firms: int,
    n_years: int,
    seed: int,
    solve_maxiter: int = 1000,
) -> np.ndarray:
    _, _, _, _, ppi, opi, _, converged = solve_model(params, maxiter=solve_maxiter)
    if not converged:
        raise RuntimeError("solve_model did not converge")
    m_tilde_nt = _simulate_all_moments(params, ppi, opi, n_firms, n_years, seed)
    return np.array([m_tilde_nt[name] for name in MOMENT_NAMES], dtype=float)


def compute_full_ii_jacobian(
    params_base: Parameters,
    n_firms: int = 5000,
    n_years: int = 20,
    seed: int = 212311,
    solve_maxiter: int = 1000,
) -> np.ndarray:
    theta0 = _full_ii_parameter_vector(params_base)
    step = np.array([max(abs(theta0[i]) * 1e-4, 1e-6) for i in range(7)], dtype=float)
    step[3] = max(step[3], 1e-5)
    step[6] = max(step[6], 1e-5)

    g = np.empty((7, 7), dtype=float)
    for j in range(7):
        theta_plus = theta0.copy()
        theta_minus = theta0.copy()
        h = step[j]
        theta_plus[j] += h
        theta_minus[j] -= h

        m_plus = _full_ii_moment_vector_from_params(
            _full_ii_params_from_vector(params_base, theta_plus),
            n_firms=n_firms,
            n_years=n_years,
            seed=seed,
            solve_maxiter=solve_maxiter,
        )
        m_minus = _full_ii_moment_vector_from_params(
            _full_ii_params_from_vector(params_base, theta_minus),
            n_firms=n_firms,
            n_years=n_years,
            seed=seed,
            solve_maxiter=solve_maxiter,
        )
        g[:, j] = (m_plus - m_minus) / (2.0 * h)

    return g


def compute_full_ii_asymptotic_variance(
    params_base: Parameters,
    w: np.ndarray,
    n_firms: int = 5000,
    n_years: int = 20,
    seed: int = 212311,
    solve_maxiter: int = 1000,
    sample_size: int = 1,
) -> dict[str, Any]:
    gf = compute_full_ii_jacobian(
        params_base,
        n_firms=n_firms,
        n_years=n_years,
        seed=seed,
        solve_maxiter=solve_maxiter,
    )
    wf = np.asarray(w, dtype=float)
    avar = np.linalg.inv(gf @ wf @ gf.T)
    vcov = avar / float(sample_size)
    se = np.sqrt(np.diag(vcov))

    return {
        "G": gf,
        "avar": avar,
        "vcov": vcov,
        "se": se,
        "parameter_names": ["γ", "μη", "ση2", "ρω", "σν2", "ϵ", "δ"],
    }


def select_best_grid_start(df_grid: pd.DataFrame, target_moments: dict[str, float], w: np.ndarray) -> dict[str, float | int]:
    m_hat = _full_ii_target_vector(target_moments)
    best_idx = -1
    best_obj = np.inf

    failed_col = _col(df_grid, ["failed"])

    for i in range(len(df_grid)):
        if bool(failed_col.iloc[i]):
            continue

        m_tilde = np.array([
            float(_col(df_grid, ["avg_isr"]).iloc[i]),
            float(_col(df_grid, ["var_log1p_isr"]).iloc[i]),
            float(_col(df_grid, ["avg_gross_margin"]).iloc[i]),
            float(_col(df_grid, ["γ_OLS"]).iloc[i]),
            float(_col(df_grid, ["ρ_ω", "ρω"]).iloc[i]),
            float(_col(df_grid, ["σ_η2", "ση2_aux"]).iloc[i]),
            float(_col(df_grid, ["μ_η", "μη_aux"]).iloc[i]),
        ])
        if not np.all(np.isfinite(m_tilde)):
            continue

        m = m_hat - m_tilde
        obj = float(m @ (w @ m))
        if obj < best_obj:
            best_obj = obj
            best_idx = i

    if best_idx < 0:
        raise RuntimeError("No valid candidate rows found in df_grid")

    return {
        "row_index": int(best_idx),
        "obj_value": float(best_obj),
        "γ": float(_col(df_grid, ["γ", "gamma"]).iloc[best_idx]),
        "μη": float(_col(df_grid, ["μη", "mu_eta"]).iloc[best_idx]),
        "ση2": float(_col(df_grid, ["ση2", "sigma_eta2"]).iloc[best_idx]),
        "ρω": float(_col(df_grid, ["ρω", "ρ_ω", "rho_omega"]).iloc[best_idx]),
        "σν2": float(_col(df_grid, ["σν2", "sigma_nu2"]).iloc[best_idx]),
        "ϵ": float(_col(df_grid, ["ϵ", "epsilon"]).iloc[best_idx]),
        "δ": float(_col(df_grid, ["δ", "delta"]).iloc[best_idx]),
    }


def estimate_params_ii_full(
    target_moments: dict[str, float],
    init_guess: np.ndarray,
    w: np.ndarray,
    n_firms: int = 200,
    n_years: int = 50,
    gamma_lb: float = 0.05,
    gamma_ub: float = 3.0,
    mu_eta_lb: float = -5.0,
    mu_eta_ub: float = 5.0,
    sigma2_lb: float = 1e-6,
    sigma2_ub: float = 5.0,
    rho_lb: float = -0.999,
    rho_ub: float = 0.999,
    sigma_nu2_lb: float = 1e-6,
    sigma_nu2_ub: float = 5.0,
    epsilon_lb: float = 1.1,
    epsilon_ub: float = 20.0,
    delta_lb: float = 0.001,
    delta_ub: float = 0.999,
    seed: int = 212311,
    max_iter: int = 1000,
    verbose: bool = True,
    g_abstol: float = 1e-4,
) -> dict[str, Any]:
    m_hat = _full_ii_target_vector(target_moments)

    if verbose:
        print("\n=== Full II Estimation - Data Moments ===")
        for i, name in enumerate(MOMENT_NAMES):
            print(f"  {name:16s} = {m_hat[i]:10.6f}")
        print("\nStarting Nelder-Mead optimization")

    def unpack(x: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
        gamma_n = float(np.clip(x[0], gamma_lb, gamma_ub))
        mu_eta_n = float(np.clip(x[1], mu_eta_lb, mu_eta_ub))
        sigma_eta2_n = float(np.clip(np.exp(x[2]), sigma2_lb, sigma2_ub))
        rho_n = float(np.clip(np.tanh(x[3]), rho_lb, rho_ub))
        sigma_nu2_n = float(np.clip(np.exp(x[4]), sigma_nu2_lb, sigma_nu2_ub))
        epsilon_n = float(np.clip(x[5], epsilon_lb, epsilon_ub))
        delta_n = float(np.clip(1.0 / (1.0 + np.exp(-x[6])), delta_lb, delta_ub))
        return gamma_n, mu_eta_n, sigma_eta2_n, rho_n, sigma_nu2_n, epsilon_n, delta_n

    iter_count = {"n": 0}

    def obj(x: np.ndarray) -> float:
        iter_count["n"] += 1
        gamma_n, mu_eta_n, sigma_eta2_n, rho_n, sigma_nu2_n, epsilon_n, delta_n = unpack(x)

        try:
            params_iter = Parameters(
                mu_eta=mu_eta_n,
                sigma_eta2=sigma_eta2_n,
                rho_omega=rho_n,
                gamma=gamma_n,
                delta=delta_n,
                epsilon=epsilon_n,
                sigma_nu2=sigma_nu2_n,
            )
            _, _, _, _, ppi, opi, _, _ = solve_model(params_iter)
            m_tilde_nt = _simulate_all_moments(params_iter, ppi, opi, n_firms, n_years, seed)
            m_tilde = np.array([m_tilde_nt[name] for name in MOMENT_NAMES], dtype=float)
            m = m_hat - m_tilde
            sse = float(m @ (w @ m))

            if verbose:
                print(f"iter {iter_count['n']:4d} obj={sse:12.6f} moments={np.round(m_tilde, 5)}")
            return sse
        except Exception:
            if verbose:
                print(f"iter {iter_count['n']:4d} model failed, penalty returned")
            return 1e10

    init_guess = np.asarray(init_guess, dtype=float)
    if init_guess.size != 7:
        raise ValueError("init_guess must have length 7 ordered [gamma, mu_eta, sigma_eta2, rho, sigma_nu2, epsilon, delta]")

    gamma_init, mu_eta_init, sigma_eta2_init, rho_init, sigma_nu2_init, epsilon_init, delta_init = init_guess
    x0 = np.array(
        [
            np.clip(gamma_init, gamma_lb, gamma_ub),
            np.clip(mu_eta_init, mu_eta_lb, mu_eta_ub),
            np.log(np.clip(sigma_eta2_init, sigma2_lb, sigma2_ub)),
            np.arctanh(np.clip(rho_init, rho_lb, rho_ub)),
            np.log(np.clip(sigma_nu2_init, sigma_nu2_lb, sigma_nu2_ub)),
            np.clip(epsilon_init, epsilon_lb, epsilon_ub),
            np.log(np.clip(delta_init, delta_lb, delta_ub) / (1.0 - np.clip(delta_init, delta_lb, delta_ub))),
        ],
        dtype=float,
    )

    res = minimize(
        obj,
        x0,
        method="Nelder-Mead",
        options={"maxiter": int(max_iter), "xatol": 1e-4, "fatol": float(g_abstol), "disp": False},
    )

    gamma_est, mu_eta_est, sigma_eta2_est, rho_est, sigma_nu2_est, epsilon_est, delta_est = unpack(res.x)

    if verbose:
        print("\n=== Full II Estimation Complete ===")
        print(f"Converged : {res.success}")
        print(f"gamma     : {gamma_est:10.6f}")
        print(f"mu_eta    : {mu_eta_est:10.6f}")
        print(f"sigma_eta2: {sigma_eta2_est:10.6f}")
        print(f"rho_omega : {rho_est:10.6f}")
        print(f"sigma_nu2 : {sigma_nu2_est:10.6f}")
        print(f"epsilon   : {epsilon_est:10.6f}")
        print(f"delta     : {delta_est:10.6f}")
        print(f"Objective : {res.fun:.8f}")

    return {
        "γ": gamma_est,
        "μη": mu_eta_est,
        "ση2": sigma_eta2_est,
        "ρω": rho_est,
        "σν2": sigma_nu2_est,
        "ϵ": epsilon_est,
        "δ": delta_est,
        "obj_value": float(res.fun),
        "result": res,
    }
