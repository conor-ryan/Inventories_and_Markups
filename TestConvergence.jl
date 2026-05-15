using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Plots, Interpolations, LineSearch, Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf, BenchmarkTools, Profile

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

# ---------------------------------------------------------------------------
# Parameter draws — same bounds and transforms as SimulateMoments.jl
# ---------------------------------------------------------------------------
ϵ_bounds   = (4.0, 20.0)
σν2_bounds = (0.001, 0.8)
δ_bounds   = (0.001, 0.2)
μω_bounds  = (0.001, 0.1)
γ_bounds   = (0.7, 1.5)
ση2_bounds = (0.01, 0.1)
ρ_bounds   = (0.05, 0.9)

n_param_points = 50
param_bounds = [
    γ_bounds,
    μω_bounds,
    ση2_bounds,
    ρ_bounds,
    σν2_bounds,
    ϵ_bounds,
    δ_bounds
]

param_vectors = halton_param_vectors(param_bounds, n_param_points; seed=212311)

# Recover μη = log(μω) * (1 - ρ) and δ from log space
for v in param_vectors
    v[2] = log(v[2]) * (1.0 - v[4])
end

# ---------------------------------------------------------------------------
# Scenario settings
# ---------------------------------------------------------------------------
# A: reference — value function convergence, tight tolerance, fine grid
NS_A   = 1200
TOL_A  = 1e-2
CONV_A = :policy
MAXITER_A = 5_000

# B: fast — policy function convergence, coarse tolerance, coarser grid
NS_B   = 400
TOL_B  = 1e-2
CONV_B = :policy
MAXITER_B = 5_000

# Simulation settings (same for both scenarios)
N_FIRMS = 500
N_YEARS = 20
SEED    = 212311

moment_keys = [:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :avg_opex_sales]
n_moments   = length(moment_keys)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
diffs = fill(NaN, n_param_points, n_moments)   # diffs[i, k] = m_A[k] - m_B[k]

@printf("\nRunning convergence test on %d parameter draws...\n", n_param_points)

for (i, v) in enumerate(param_vectors)
    γ_i, μη_i, ση2_i, ρ_i, σν2_i, ϵ_i, δ_i = v

    # Skip draws that would produce degenerate parameters
    if ση2_i <= 0.0 || σν2_i <= 0.0 || δ_i <= 0.0 || δ_i >= 1.0
        @printf("  [%3d] skipped (degenerate params)\n", i)
        continue
    end

    local m_A, m_B
    try
        params_A = Parameters(c=1.0, fc=0.0, μη=μη_i, ση2=ση2_i, ρ_ω=ρ_i,
                              γ=γ_i, δ=δ_i, β=0.95, ϵ=ϵ_i, μν=1.0, σν2=σν2_i,
                              Ns=NS_A, scale=1.0, size=100.0)
        _, n_pol_A, p_pol_A, _, conv_A = solve_value_function(params_A;
            conv=CONV_A, tol=TOL_A, maxiter=MAXITER_A, fast_interp=true)
        if !conv_A
            @printf("  [%3d] skipped (reference model did not converge)\n", i)
            continue
        end
        if any(sum(n_pol_A,dims=1) .== 0)
            @printf("  [%3d] skipped (reference model has zero policy in some states)\n", i)
            continue
        end
        ppi_nodes_A, opi_nodes_A = build_fast_policy_interpolants(params_A.Sgrid, p_pol_A, n_pol_A)
        ppi_A = OmegaPolicyInterp(ppi_nodes_A, params_A.ω_grid)
        opi_A = OmegaPolicyInterp(opi_nodes_A, params_A.ω_grid)
        m_A   = _simulate_all_moments(params_A, ppi_A, opi_A, N_FIRMS, N_YEARS, SEED)

        params_B = Parameters(c=1.0, fc=0.0, μη=μη_i, ση2=ση2_i, ρ_ω=ρ_i,
                              γ=γ_i, δ=δ_i, β=0.95, ϵ=ϵ_i, μν=1.0, σν2=σν2_i,
                              Ns=NS_B, scale=1.0, size=100.0)
        _, n_pol_B, p_pol_B, _, _ = solve_value_function(params_B;
            conv=CONV_B, tol=TOL_B, maxiter=MAXITER_B, fast_interp=true)
        ppi_nodes_B, opi_nodes_B = build_fast_policy_interpolants(params_B.Sgrid, p_pol_B, n_pol_B)
        ppi_B = OmegaPolicyInterp(ppi_nodes_B, params_B.ω_grid)
        opi_B = OmegaPolicyInterp(opi_nodes_B, params_B.ω_grid)
        m_B   = _simulate_all_moments(params_B, ppi_B, opi_B, N_FIRMS, N_YEARS, SEED)

        for (k, key) in enumerate(moment_keys)
            diffs[i, k] = getfield(m_A, key) - getfield(m_B, key)
        end
        @printf("  [%3d] done\n", i)
    catch e
        @printf("  [%3d] error: %s\n", i, string(e))
    end
end

# ---------------------------------------------------------------------------
# Order statistics on differences
# ---------------------------------------------------------------------------
println("\n" * "="^80)
println("Convergence comparison: scenario A (value conv, tol=$(TOL_A), Ns=$(NS_A))")
println("                    vs  scenario B (policy conv, tol=$(TOL_B), Ns=$(NS_B))")
println("Differences = A - B across $(n_param_points) parameter draws")
println("="^80)
@printf("%-22s  %10s  %10s  %10s  %10s  %10s  %10s\n",
        "moment", "p10", "p25", "median", "p75", "p90", "max|diff|")
println("-"^86)
for (k, key) in enumerate(moment_keys)
    col  = diffs[:, k]
    vals = filter(isfinite, col)
    if isempty(vals)
        @printf("%-22s  (no valid draws)\n", key)
        continue
    end
    @printf("%-22s  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
            key,
            quantile(vals, 0.10),
            quantile(vals, 0.25),
            quantile(vals, 0.50),
            quantile(vals, 0.75),
            quantile(vals, 0.90),
            maximum(abs.(vals)))
end
println("="^80)
