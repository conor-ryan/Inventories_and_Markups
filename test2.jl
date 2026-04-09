using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

function simulate_moments_mc(params::Parameters;
                             n_firms::Int,
                             n_months::Int,
                             n_replications::Int,
                             base_seed::Int,
                             solve_maxiter::Int=1000)
    n_months % 12 == 0 || error("n_months must be divisible by 12 so annual moments are well defined")
    n_years = div(n_months, 12)

    println("Solving model once...")
    _, _, _, _, ppi, opi, _, converged = solve_model(params; maxiter=solve_maxiter)
    converged || error("solve_model did not converge")

    println("Running Monte Carlo replications...")
    draws = Vector{NamedTuple}(undef, n_replications)
    for rep in 1:n_replications
        println(rep)
        seed = base_seed + rep - 1
        draws[rep] = _simulate_all_moments(params, ppi, opi, n_firms, n_years, seed)
    end

    return draws
end

function summarize_draws(draws::Vector{NamedTuple})
    moment_names = (:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :μ_η)
    summary = NamedTuple[]

    for name in moment_names
        values = [getfield(draw, name) for draw in draws]
        push!(summary, (
            moment=name,
            mean=mean(values),
            variance=var(values)
        ))
    end

    return summary
end

function print_summary(summary::Vector{NamedTuple};
                       n_firms::Int,
                       n_months::Int,
                       n_replications::Int,
                       base_seed::Int)
    println("Monte Carlo configuration:")
    @printf("  firms        = %d\n", n_firms)
    @printf("  months       = %d\n", n_months)
    @printf("  replications = %d\n", n_replications)
    @printf("  base_seed    = %d\n", base_seed)

    println("\nmoment, mean, variance")
    for row in summary
        @printf("%s, %.6f, %.6f\n", String(row.moment), row.mean, row.variance)
    end
end

params = Parameters(c=1.0, fc=0.0, μη=log(0.05), ση2=0.05, ρ_ω=0.2, γ=0.9,
                    δ=0.01, β=0.95, ϵ=6.0, μν=1, σν2=0.09,
                    Smax=30, Ns=200, scale=1.0, size=100)

n_firms = 5000
n_months = 240
n_replications = 50
base_seed = 212311

draws = simulate_moments_mc(params;
                            n_firms=n_firms,
                            n_months=n_months,
                            n_replications=n_replications,
                            base_seed=base_seed)

summary = summarize_draws(draws)
print_summary(summary;
              n_firms=n_firms,
              n_months=n_months,
              n_replications=n_replications,
              base_seed=base_seed)
