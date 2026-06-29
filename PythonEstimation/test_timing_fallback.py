"""test_timing_fallback.py

Benchmarks the cost of the sign-change pre-check and proxy-profit fallback.

Q1: When the fallback is NOT triggered, does the pre-check slow things down?
Q2: When the fallback IS triggered, how much time does it add?
    (Tested for both: fallback succeeds and fallback hard-fails.)

Run from Code/PythonEstimation/:
    python test_timing_fallback.py
"""
import math
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtri

from model_functions import (
    Parameters, price_residual, proxy_profit, limit_price,
    _brent_root_price, _maximize_proxy_profit_jit,
    _price_residual_jit, _proxy_profit_jit,
)

P_HI  = 50.0
REPS  = 500   # per-state repetitions
SREPS = 10    # full-sweep repetitions


def mean_us(fn, reps=REPS):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6  # microseconds


def make_params(**kw):
    defaults = dict(sigma_eta2=0.2, rho_omega=0.1, sigma_nu2=0.1,
                    delta=0.05, ns=200, q=15, q_omega=5)
    defaults.update(kw)
    return Parameters(**defaults)


def nu_threshold(par):
    return math.exp(par.mu_log_nu + par.sigma_log_nu * float(ndtri(1e-8)))


def p_lo_for(par, s, omega):
    p_unc = limit_price(par.mu_log_nu, par.sigma_log_nu,
                        par.epsilon, par.gamma, par.c, omega)
    return max(0.99 * p_unc, (nu_threshold(par) / s) ** (1.0 / par.epsilon)), p_unc


# ── Scenario parameters ────────────────────────────────────────────────────────
# A: eps*(1-gam) = 0.4  → Brent always succeeds
par_A = make_params(epsilon=4.0, gamma=0.9,  mu_eta=-1.0)

# B: eps*(1-gam) ~ 1.65 → Brent fails, fallback finds profit > 0
#    (reproduces case 45 from the comparison test: rng seed 42, draw 25)
rng = np.random.default_rng(42)
for _ in range(26):
    eps_r  = rng.uniform(4.0, 20.0)
    gam_r  = rng.uniform(0.5,  0.9)
    mu_w_r = rng.uniform(0.01, 0.2)
    rho_r  = rng.uniform(0.0,  0.3)
    mu_e_r = math.log(mu_w_r) * (1.0 - rho_r)
    sig2_r = rng.uniform(0.1,  0.5)
    snu2_r = rng.uniform(0.01, 0.3)
    dlt_r  = rng.uniform(0.01, 0.1)
par_B = make_params(epsilon=eps_r, gamma=gam_r, mu_eta=mu_e_r,
                    sigma_eta2=sig2_r, rho_omega=rho_r, sigma_nu2=snu2_r, delta=dlt_r)

# C: eps*(1-gam) = 4.0  → Brent fails, proxy_profit < 0 everywhere (hard fail)
par_C = make_params(epsilon=8.0, gamma=0.5, mu_eta=-1.0)


# ── JIT warm-up ───────────────────────────────────────────────────────────────
print("Warming up JIT compilation (first call compiles, subsequent calls are timed)...")
for par in [par_A, par_B, par_C]:
    c = par.c
    omega = par.omega_grid[par.q_omega // 2]
    s = par.sgrid[100]
    plo, _ = p_lo_for(par, s, omega)
    _price_residual_jit(plo, s, c, omega, par.epsilon, par.gamma,
                        par.mu_log_nu, par.sigma_log_nu, par.gl_nodes, par.gl_weights)
    _brent_root_price(s, c, omega, plo, P_HI, par.epsilon, par.gamma,
                      par.mu_log_nu, par.sigma_log_nu, par.gl_nodes, par.gl_weights)
    _maximize_proxy_profit_jit(s, c, omega, plo, P_HI, par.epsilon, par.gamma,
                               par.mu_log_nu, par.sigma_log_nu, par.gl_nodes, par.gl_weights)
print("Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Q1: Pre-check overhead in the normal path (Brent succeeds)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Q1: Pre-check overhead — normal path (Brent succeeds every state)")
print(f"    params: eps={par_A.epsilon}, gam={par_A.gamma},",
      f"eps*(1-gam)={par_A.epsilon*(1-par_A.gamma):.2f}")
print("=" * 65)

c = par_A.c
omega = par_A.omega_grid[par_A.q_omega // 2]
s = par_A.sgrid[100]
plo, punc = p_lo_for(par_A, s, omega)

# Verify Brent succeeds here
rlo = price_residual(plo, s, c, omega, par_A)
rhi = price_residual(P_HI, s, c, omega, par_A)
assert rlo * rhi <= 0.0, "Brent should succeed for par_A"

# Python: cost of the 2 pre-check evaluations
t_py_precheck = mean_us(lambda: (
    price_residual(plo, s, c, omega, par_A),
    price_residual(P_HI, s, c, omega, par_A),
))

# Python: cost of a successful brentq call (what follows the pre-check)
t_py_brent_ok = mean_us(
    lambda: brentq(lambda p: price_residual(p, s, c, omega, par_A),
                   plo, P_HI, xtol=1e-12)
)

# JIT: cost of 2 _price_residual_jit evaluations
t_jit_precheck = mean_us(lambda: (
    _price_residual_jit(plo, s, c, omega, par_A.epsilon, par_A.gamma,
                        par_A.mu_log_nu, par_A.sigma_log_nu,
                        par_A.gl_nodes, par_A.gl_weights),
    _price_residual_jit(P_HI, s, c, omega, par_A.epsilon, par_A.gamma,
                        par_A.mu_log_nu, par_A.sigma_log_nu,
                        par_A.gl_nodes, par_A.gl_weights),
))

# JIT: cost of a successful _brent_root_price call
t_jit_brent_ok = mean_us(
    lambda: _brent_root_price(s, c, omega, plo, P_HI,
                              par_A.epsilon, par_A.gamma,
                              par_A.mu_log_nu, par_A.sigma_log_nu,
                              par_A.gl_nodes, par_A.gl_weights)
)

print(f"\n  Python path (solve_price_policy):")
print(f"    2 x price_residual (pre-check):    {t_py_precheck:7.2f} us")
print(f"    brentq call (succeeds):            {t_py_brent_ok:7.2f} us")
print(f"    overhead fraction:                 {100*t_py_precheck/t_py_brent_ok:6.1f}%")

print(f"\n  JIT path (_price_policy_sweep):")
print(f"    2 x _price_residual_jit (check):   {t_jit_precheck:7.2f} us")
print(f"    _brent_root_price (succeeds):      {t_jit_brent_ok:7.2f} us")
print(f"    overhead fraction:                 {100*t_jit_precheck/t_jit_brent_ok:6.1f}%")
print(f"\n  Note: in _price_policy_sweep the 2 endpoint evals were ALREADY inside")
print(f"  _brent_root_price before our change, so the JIT path has zero new overhead.")


# ══════════════════════════════════════════════════════════════════════════════
# Q2a: Fallback cost — fallback SUCCEEDS (profit > 0 found)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("Q2a: Fallback cost — fallback SUCCEEDS (profit > 0)")
print(f"     params: eps={par_B.epsilon:.2f}, gam={par_B.gamma:.2f},",
      f"eps*(1-gam)={par_B.epsilon*(1-par_B.gamma):.2f}")
print("=" * 65)

c = par_B.c
omega = par_B.omega_grid[par_B.q_omega // 2]
# Find a state where Brent fails and proxy_profit > 0
s_B = None
for s_candidate in par_B.sgrid[10:]:
    plo_c, _ = p_lo_for(par_B, s_candidate, omega)
    rlo = price_residual(plo_c, s_candidate, c, omega, par_B)
    rhi = price_residual(P_HI,  s_candidate, c, omega, par_B)
    if rlo * rhi > 0.0:
        res = minimize_scalar(lambda p: -proxy_profit(p, s_candidate, c, omega, par_B),
                              bounds=(plo_c, P_HI), method='bounded')
        if proxy_profit(res.x, s_candidate, c, omega, par_B) > 0.0:
            s_B = s_candidate
            plo_B = plo_c
            break

if s_B is None:
    print("  Could not find a fallback-success state for par_B; skipping Q2a.")
else:
    rlo = price_residual(plo_B, s_B, c, omega, par_B)
    rhi = price_residual(P_HI,  s_B, c, omega, par_B)
    res_B = minimize_scalar(lambda p: -proxy_profit(p, s_B, c, omega, par_B),
                            bounds=(plo_B, P_HI), method='bounded')
    print(f"\n  Verify: r_lo={rlo:.3f}, r_hi={rhi:.3f}  (same sign -> Brent fails)")
    print(f"  Verify: profit at fallback opt = {proxy_profit(res_B.x, s_B, c, omega, par_B):.6f}  (>0)")

    # Old code path: brentq raises ValueError (iterates internally before giving up)
    def _py_old_B():
        try:
            brentq(lambda p: price_residual(p, s_B, c, omega, par_B),
                   plo_B, P_HI, xtol=1e-12)
        except ValueError:
            pass

    # New code path: pre-check detects no sign change, jumps straight to fallback
    def _py_new_B():
        r0 = price_residual(plo_B, s_B, c, omega, par_B)
        r1 = price_residual(P_HI,  s_B, c, omega, par_B)
        if r0 * r1 > 0.0:
            minimize_scalar(lambda p: -proxy_profit(p, s_B, c, omega, par_B),
                            bounds=(plo_B, P_HI), method='bounded')

    t_py_old_B = mean_us(_py_old_B)
    t_py_new_B = mean_us(_py_new_B)

    # JIT: old path = _brent_root_price (fails fast after 2 evals)
    def _jit_old_B():
        _brent_root_price(s_B, c, omega, plo_B, P_HI,
                          par_B.epsilon, par_B.gamma,
                          par_B.mu_log_nu, par_B.sigma_log_nu,
                          par_B.gl_nodes, par_B.gl_weights)

    # JIT: new path = failing Brent + proxy-profit maximize
    def _jit_new_B():
        _brent_root_price(s_B, c, omega, plo_B, P_HI,
                          par_B.epsilon, par_B.gamma,
                          par_B.mu_log_nu, par_B.sigma_log_nu,
                          par_B.gl_nodes, par_B.gl_weights)
        _maximize_proxy_profit_jit(s_B, c, omega, plo_B, P_HI,
                                   par_B.epsilon, par_B.gamma,
                                   par_B.mu_log_nu, par_B.sigma_log_nu,
                                   par_B.gl_nodes, par_B.gl_weights)

    t_jit_old_B = mean_us(_jit_old_B)
    t_jit_new_B = mean_us(_jit_new_B)

    print(f"\n  Python path:")
    print(f"    old (brentq -> ValueError):        {t_py_old_B:7.2f} us")
    print(f"    new (precheck + minimize_scalar):  {t_py_new_B:7.2f} us")
    print(f"    added cost:                        {t_py_new_B - t_py_old_B:+7.2f} us",
          f"  ({100*(t_py_new_B/t_py_old_B - 1):+.0f}%)")

    print(f"\n  JIT path:")
    print(f"    old (Brent fails after 2 evals):   {t_jit_old_B:7.2f} us")
    print(f"    new (+ _maximize_proxy_profit):    {t_jit_new_B:7.2f} us")
    print(f"    added cost:                        {t_jit_new_B - t_jit_old_B:+7.2f} us",
          f"  ({100*(t_jit_new_B/t_jit_old_B - 1):+.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Q2b: Fallback cost — fallback HARD-FAILS (proxy_profit < 0)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("Q2b: Fallback cost — fallback HARD-FAILS (proxy_profit < 0)")
print(f"     params: eps={par_C.epsilon}, gam={par_C.gamma},",
      f"eps*(1-gam)={par_C.epsilon*(1-par_C.gamma):.2f}")
print("=" * 65)

c = par_C.c
omega = par_C.omega_grid[par_C.q_omega // 2]
s_C = par_C.sgrid[100]
plo_C, punc_C = p_lo_for(par_C, s_C, omega)

rlo = price_residual(plo_C, s_C, c, omega, par_C)
rhi = price_residual(P_HI,  s_C, c, omega, par_C)
res_C = minimize_scalar(lambda p: -proxy_profit(p, s_C, c, omega, par_C),
                        bounds=(plo_C, P_HI), method='bounded')
profit_C = proxy_profit(res_C.x, s_C, c, omega, par_C)
print(f"\n  Verify: r_lo={rlo:.3f}, r_hi={rhi:.3f}  (same sign -> Brent fails)")
print(f"  Verify: profit at fallback opt = {profit_C:.6f}  (<0 -> hard fail)")

def _py_old_C():
    try:
        brentq(lambda p: price_residual(p, s_C, c, omega, par_C),
               plo_C, P_HI, xtol=1e-12)
    except ValueError:
        pass

def _py_new_C():
    r0 = price_residual(plo_C, s_C, c, omega, par_C)
    r1 = price_residual(P_HI,  s_C, c, omega, par_C)
    if r0 * r1 > 0.0:
        minimize_scalar(lambda p: -proxy_profit(p, s_C, c, omega, par_C),
                        bounds=(plo_C, P_HI), method='bounded')

def _jit_old_C():
    _brent_root_price(s_C, c, omega, plo_C, P_HI,
                      par_C.epsilon, par_C.gamma,
                      par_C.mu_log_nu, par_C.sigma_log_nu,
                      par_C.gl_nodes, par_C.gl_weights)

def _jit_new_C():
    _brent_root_price(s_C, c, omega, plo_C, P_HI,
                      par_C.epsilon, par_C.gamma,
                      par_C.mu_log_nu, par_C.sigma_log_nu,
                      par_C.gl_nodes, par_C.gl_weights)
    _maximize_proxy_profit_jit(s_C, c, omega, plo_C, P_HI,
                               par_C.epsilon, par_C.gamma,
                               par_C.mu_log_nu, par_C.sigma_log_nu,
                               par_C.gl_nodes, par_C.gl_weights)

t_py_old_C  = mean_us(_py_old_C)
t_py_new_C  = mean_us(_py_new_C)
t_jit_old_C = mean_us(_jit_old_C)
t_jit_new_C = mean_us(_jit_new_C)

print(f"\n  Python path:")
print(f"    old (brentq -> ValueError):        {t_py_old_C:7.2f} us")
print(f"    new (precheck + minimize_scalar):  {t_py_new_C:7.2f} us")
print(f"    added cost:                        {t_py_new_C - t_py_old_C:+7.2f} us",
      f"  ({100*(t_py_new_C/t_py_old_C - 1):+.0f}%)")

print(f"\n  JIT path:")
print(f"    old (Brent fails after 2 evals):   {t_jit_old_C:7.2f} us")
print(f"    new (+ _maximize_proxy_profit):    {t_jit_new_C:7.2f} us")
print(f"    added cost:                        {t_jit_new_C - t_jit_old_C:+7.2f} us",
      f"  ({100*(t_jit_new_C/t_jit_old_C - 1):+.0f}%)")

print()
print("  Note: hard-fail cost == fallback-success cost — the optimizer runs to")
print("  convergence either way; only the subsequent profit check differs.")


# ══════════════════════════════════════════════════════════════════════════════
# Full solve_price_policy sweep (Python path, all ns states)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("Full solve_price_policy sweep over all inventory states (Python)")
print("=" * 65)


def sweep_old(par, omega):
    """Original behavior: brentq with try/except, no pre-check."""
    c = par.c
    nu_thr = nu_threshold(par)
    p_unc = limit_price(par.mu_log_nu, par.sigma_log_nu,
                        par.epsilon, par.gamma, c, omega)
    p_lo = 0.99 * p_unc
    for s in par.sgrid:
        if s == 0.0:
            continue
        plo_s = max(p_lo, (nu_thr / s) ** (1.0 / par.epsilon))
        try:
            brentq(lambda p: price_residual(p, s, c, omega, par),
                   plo_s, P_HI, xtol=1e-12)
        except ValueError:
            pass


def sweep_new(par, omega):
    """New behavior: pre-check, then brentq or minimize_scalar fallback."""
    c = par.c
    nu_thr = nu_threshold(par)
    p_unc = limit_price(par.mu_log_nu, par.sigma_log_nu,
                        par.epsilon, par.gamma, c, omega)
    p_lo = 0.99 * p_unc
    for s in par.sgrid:
        if s == 0.0:
            continue
        plo_s = max(p_lo, (nu_thr / s) ** (1.0 / par.epsilon))
        r0 = price_residual(plo_s, s, c, omega, par)
        r1 = price_residual(P_HI,  s, c, omega, par)
        if r0 * r1 <= 0.0:
            try:
                brentq(lambda p: price_residual(p, s, c, omega, par),
                       plo_s, P_HI, xtol=1e-12)
            except ValueError:
                pass
        else:
            minimize_scalar(lambda p: -proxy_profit(p, s, c, omega, par),
                            bounds=(plo_s, P_HI), method='bounded')


scenarios = [
    ("Normal path  eps*(1-gam)=0.40  (Brent works)", par_A),
    ("Hard fail    eps*(1-gam)=4.00  (fb hard-fail)", par_C),
]
if s_B is not None:
    scenarios.insert(1, ("Fallback succ eps*(1-gam)={:.2f}  (fb helps)".format(
        par_B.epsilon * (1 - par_B.gamma)), par_B))

for label, par in scenarios:
    omega = par.omega_grid[par.q_omega // 2]

    t0 = time.perf_counter()
    for _ in range(SREPS):
        sweep_old(par, omega)
    t_old_ms = (time.perf_counter() - t0) / SREPS * 1e3

    t0 = time.perf_counter()
    for _ in range(SREPS):
        sweep_new(par, omega)
    t_new_ms = (time.perf_counter() - t0) / SREPS * 1e3

    print(f"\n  {label}")
    print(f"    old: {t_old_ms:6.1f} ms   new: {t_new_ms:6.1f} ms",
          f"  delta: {t_new_ms - t_old_ms:+5.1f} ms  ({100*(t_new_ms/t_old_ms - 1):+.0f}%)")
