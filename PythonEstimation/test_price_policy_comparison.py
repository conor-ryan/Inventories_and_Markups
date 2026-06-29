"""test_price_policy_comparison.py

Compares three price-policy strategies over a sample of parameter vectors:
  A: Brent root-finding only — falls back to p_unc when no sign change
     (replicates original behavior before the fallback was added)
  B: Brent with proxy-profit fallback — the updated production behavior
  C: Proxy-profit direct maximization only

Checks that B and C agree wherever both find a valid price, and that A
diverges from C in cases where epsilon*(1-gamma) > 1 forces Brent to fail.

Run from Code/PythonEstimation/:
    python test_price_policy_comparison.py
"""

import math
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtri

from model_functions import Parameters, price_residual, proxy_profit, limit_price

P_HI = 50.0


# ── Three solver implementations ──────────────────────────────────────────────

def _solve_A_brent_only(params, c_tilde, omega):
    """Brent root-finding, no fallback (original behavior)."""
    p_unc = limit_price(params.mu_log_nu, params.sigma_log_nu,
                        params.epsilon, params.gamma, c_tilde, omega)
    p_lo = 0.99 * p_unc
    nu_threshold = math.exp(params.mu_log_nu + params.sigma_log_nu * float(ndtri(1e-8)))
    p_policy = np.empty(params.ns)
    n_failed = 0
    for i, s in enumerate(params.sgrid):
        if s == 0.0:
            p_policy[i] = 1.0
            continue
        p_lo_s = max(p_lo, (nu_threshold / s) ** (1.0 / params.epsilon))
        try:
            p_opt = brentq(
                lambda p: price_residual(p, s, c_tilde, omega, params),
                p_lo_s, P_HI, xtol=1e-12,
            )
            if proxy_profit(p_opt, s, c_tilde, omega, params) < 0.0:
                raise ValueError
        except ValueError:
            p_opt = p_unc
            n_failed += 1
        p_policy[i] = p_opt
    return p_policy, n_failed


def _solve_B_brent_fallback(params, c_tilde, omega):
    """Brent root-finding with proxy-profit fallback (updated production behavior)."""
    p_unc = limit_price(params.mu_log_nu, params.sigma_log_nu,
                        params.epsilon, params.gamma, c_tilde, omega)
    p_lo = 0.99 * p_unc
    nu_threshold = math.exp(params.mu_log_nu + params.sigma_log_nu * float(ndtri(1e-8)))
    p_policy = np.empty(params.ns)
    n_brent = 0
    n_fallback = 0
    n_hard_fail = 0
    for i, s in enumerate(params.sgrid):
        if s == 0.0:
            p_policy[i] = 1.0
            continue
        p_lo_s = max(p_lo, (nu_threshold / s) ** (1.0 / params.epsilon))

        r_lo = price_residual(p_lo_s, s, c_tilde, omega, params)
        r_hi = price_residual(P_HI, s, c_tilde, omega, params)

        p_opt = None
        if r_lo * r_hi <= 0.0:
            try:
                p_opt = brentq(
                    lambda p: price_residual(p, s, c_tilde, omega, params),
                    p_lo_s, P_HI, xtol=1e-12,
                )
                if proxy_profit(p_opt, s, c_tilde, omega, params) < 0.0:
                    p_opt = None
                else:
                    n_brent += 1
            except ValueError:
                p_opt = None

        if p_opt is None:
            result = minimize_scalar(
                lambda p: -proxy_profit(p, s, c_tilde, omega, params),
                bounds=(p_lo_s, P_HI), method='bounded',
            )
            p_fb = result.x
            if proxy_profit(p_fb, s, c_tilde, omega, params) >= 0.0:
                p_opt = p_fb
                n_fallback += 1
            else:
                p_opt = p_unc
                n_hard_fail += 1

        p_policy[i] = p_opt
    return p_policy, n_brent, n_fallback, n_hard_fail


def _solve_C_proxy_only(params, c_tilde, omega):
    """Always maximize proxy_profit directly."""
    p_unc = limit_price(params.mu_log_nu, params.sigma_log_nu,
                        params.epsilon, params.gamma, c_tilde, omega)
    p_lo = 0.99 * p_unc
    nu_threshold = math.exp(params.mu_log_nu + params.sigma_log_nu * float(ndtri(1e-8)))
    p_policy = np.empty(params.ns)
    n_hard_fail = 0
    for i, s in enumerate(params.sgrid):
        if s == 0.0:
            p_policy[i] = 1.0
            continue
        p_lo_s = max(p_lo, (nu_threshold / s) ** (1.0 / params.epsilon))
        result = minimize_scalar(
            lambda p: -proxy_profit(p, s, c_tilde, omega, params),
            bounds=(p_lo_s, P_HI), method='bounded',
        )
        p_opt = result.x
        if proxy_profit(p_opt, s, c_tilde, omega, params) < 0.0:
            p_opt = p_unc
            n_hard_fail += 1
        p_policy[i] = p_opt
    return p_policy, n_hard_fail


# ── Parameter sample ──────────────────────────────────────────────────────────
# Structured grid covering easy (eps*(1-gamma) < 1) and hard (> 1) cases,
# plus random draws from the full estimation bounds in simulatemoments.py.

structured = []
for epsilon in [4.0, 8.0, 12.0, 20.0]:
    for gamma in [0.5, 0.6, 0.7, 0.8, 0.9]:
        structured.append((gamma, -1.0, 0.2, 0.1, 0.1, epsilon, 0.05))

rng = np.random.default_rng(42)
random_cases = []
for _ in range(100):
    eps_r   = rng.uniform(4.0, 20.0)
    gam_r   = rng.uniform(0.5, 0.9)
    mu_w_r  = rng.uniform(0.01, 0.2)
    rho_r   = rng.uniform(0.0, 0.3)
    mu_e_r  = math.log(mu_w_r) * (1.0 - rho_r)
    sig2_r  = rng.uniform(0.1, 0.5)
    snu2_r  = rng.uniform(0.01, 0.3)
    dlt_r   = rng.uniform(0.01, 0.1)
    random_cases.append((gam_r, mu_e_r, sig2_r, rho_r, snu2_r, eps_r, dlt_r))

all_cases = structured + random_cases

# ── Run comparison ────────────────────────────────────────────────────────────

HDR = (f"{'#':>3}  {'eps':>5} {'gam':>5} {'e(1-g)':>7}  "
       f"{'A_fail':>6} {'B_brnt':>7} {'B_fb':>6} {'B_hf':>5}  "
       f"{'max|A-C|':>10} {'max|B-C|':>10}  {'B=C?':>5}")
print(HDR)
print("-" * len(HDR))

summary = []
errors  = []

for idx, (gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta) in enumerate(all_cases):
    try:
        params = Parameters(
            gamma=gamma, mu_eta=mu_eta, sigma_eta2=sigma_eta2,
            rho_omega=rho_omega, sigma_nu2=sigma_nu2, epsilon=epsilon, delta=delta,
            ns=200, q=15, q_omega=5,
        )
    except Exception as exc:
        errors.append((idx, f"Parameters: {exc}"))
        continue

    c_tilde = params.c
    omega   = params.omega_grid[params.q_omega // 2]

    try:
        pol_A, n_A_fail                    = _solve_A_brent_only(params, c_tilde, omega)
        pol_B, n_B_brent, n_B_fb, n_B_hf  = _solve_B_brent_fallback(params, c_tilde, omega)
        pol_C, _                           = _solve_C_proxy_only(params, c_tilde, omega)
    except Exception as exc:
        errors.append((idx, f"Solver: {exc}"))
        continue

    eps1mg   = epsilon * (1.0 - gamma)
    # Skip s=0 (always 1.0 in all methods)
    max_AC   = float(np.max(np.abs(pol_A[1:] - pol_C[1:])))
    max_BC   = float(np.max(np.abs(pol_B[1:] - pol_C[1:])))
    agree_BC = np.allclose(pol_B[1:], pol_C[1:], atol=1e-5)

    print(f"{idx:3d}  {epsilon:5.1f} {gamma:5.2f} {eps1mg:7.3f}  "
          f"{n_A_fail:6d} {n_B_brent:7d} {n_B_fb:6d} {n_B_hf:5d}  "
          f"{max_AC:10.6f} {max_BC:10.6f}  {'YES' if agree_BC else 'NO':>5}")

    summary.append(dict(
        epsilon=epsilon, gamma=gamma, eps1mg=eps1mg,
        n_A_fail=n_A_fail, n_B_fb=n_B_fb, n_B_hf=n_B_hf,
        max_AC=max_AC, max_BC=max_BC, agree_BC=agree_BC,
    ))

# ── Summary ───────────────────────────────────────────────────────────────────
n = len(summary)
print()
print(f"{'='*60}")
print(f"Summary over {n} parameter vectors  ({len(errors)} errored out)")
print(f"  Cases with any Brent failure (A):      {sum(r['n_A_fail'] > 0 for r in summary)}")
print(f"  Cases where fallback was used (B):     {sum(r['n_B_fb']   > 0 for r in summary)}")
print(f"  Cases with hard failure (B):           {sum(r['n_B_hf']   > 0 for r in summary)}")
print(f"  B and C agree (atol=1e-5):             {sum(r['agree_BC'] for r in summary)} / {n}")

hard_cases = [r for r in summary if r['eps1mg'] > 1.0]
easy_cases = [r for r in summary if r['eps1mg'] <= 1.0]
print()
print(f"  eps*(1-gam) > 1  ({len(hard_cases)} cases):")
if hard_cases:
    print(f"    Brent failed in:  {sum(r['n_A_fail'] > 0 for r in hard_cases)}")
    print(f"    Fallback used in: {sum(r['n_B_fb']   > 0 for r in hard_cases)}")
    print(f"    B=C:              {sum(r['agree_BC'] for r in hard_cases)} / {len(hard_cases)}")
    print(f"    max |A-C| range:  [{min(r['max_AC'] for r in hard_cases):.4f}, "
          f"{max(r['max_AC'] for r in hard_cases):.4f}]")
    print(f"    max |B-C| range:  [{min(r['max_BC'] for r in hard_cases):.6f}, "
          f"{max(r['max_BC'] for r in hard_cases):.6f}]")

print(f"  eps*(1-gam) <= 1  ({len(easy_cases)} cases):")
if easy_cases:
    print(f"    Brent failed in:  {sum(r['n_A_fail'] > 0 for r in easy_cases)}")
    print(f"    Fallback used in: {sum(r['n_B_fb']   > 0 for r in easy_cases)}")
    print(f"    B=C:              {sum(r['agree_BC'] for r in easy_cases)} / {len(easy_cases)}")
    print(f"    max |A-C| range:  [{min(r['max_AC'] for r in easy_cases):.6f}, "
          f"{max(r['max_AC'] for r in easy_cases):.6f}]")

if errors:
    print()
    print("Errors:")
    for i, msg in errors:
        print(f"  case {i}: {msg}")
