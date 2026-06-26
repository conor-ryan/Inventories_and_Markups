"""Temporary timing test: solve_price_policy speedup from changes 1/2/3-lower-bound.
Delete after use.
"""
import sys
import time
import math
sys.path.insert(0, ".")

from model_functions import Parameters, solve_price_policy, solve_value_function

params = Parameters(
    c=1.0, fc=0.0,
    mu_eta=math.log(0.05),
    sigma_eta2=0.05,
    rho_omega=0.1,
    gamma=1.0,
    delta=0.05,
    beta=0.995,
    epsilon=8.0,
    mu_nu=1.0,
    sigma_nu2=0.25,
    ns=500,
    scale=1.0,
    size=100.0,
)

print(f"Grid: ns={params.ns}, n_omega={params.q_omega}")

t0 = time.perf_counter()
for j in range(params.q_omega):
    solve_price_policy(params, params.c, params.omega_grid[j])
t_price = time.perf_counter() - t0
print(f"solve_price_policy (all {params.q_omega} omega states): {t_price:.3f}s")

t0 = time.perf_counter()
sol = solve_value_function(params, tol=1e-2, maxiter=500, conv="policy")
t_total = time.perf_counter() - t0
print(f"solve_value_function total:                              {t_total:.3f}s")
print(f"  of which VFI ({sol['iterations']} iters):             {t_total - t_price:.3f}s")
print(f"solve_price_policy share: {100*t_price/t_total:.1f}%")
print(f"Converged: {sol['converged']}")
