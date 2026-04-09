using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")
include("SimulateData.jl")

function summarize_target_moments(params::Parameters;
                                  n_firms::Int,
                                  n_months::Int,
                                  burn_in::Int,
                                  seed::Int)
    df_monthly, df_annual = simulate_panel_data(params;
                                                N=n_firms,
                                                M=n_months,
                                                burn_in=burn_in,
                                                seed=seed)
    return compute_full_ii_target_moments(df_monthly, df_annual)
end

function print_moment_table(results::Vector{NamedTuple})
    println("configuration, firms, months, avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, μ_η")
    for r in results
        @printf("%s, %d, %d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n",
                r.label, r.n_firms, r.n_months,
                r.moments.avg_isr, r.moments.var_log1p_isr, r.moments.avg_gross_margin,
                r.moments.γ_OLS, r.moments.ρ_ω, r.moments.σ_η2, r.moments.μ_η)
    end
end

function print_change_table(results::Vector{NamedTuple})
    baseline = results[end].moments
    println("\ncomparison_to_largest_case, firms, months, Δavg_isr, Δvar_log1p_isr, Δavg_gross_margin, Δγ_OLS, Δρ_ω, Δσ_η2, Δμ_η")
    for r in results
        @printf("%s, %d, %d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n",
                r.label, r.n_firms, r.n_months,
                r.moments.avg_isr - baseline.avg_isr,
                r.moments.var_log1p_isr - baseline.var_log1p_isr,
                r.moments.avg_gross_margin - baseline.avg_gross_margin,
                r.moments.γ_OLS - baseline.γ_OLS,
                r.moments.ρ_ω - baseline.ρ_ω,
                r.moments.σ_η2 - baseline.σ_η2,
                r.moments.μ_η - baseline.μ_η)
    end
end

params = Parameters(c=1.0, fc=0.0, μη=log(0.05), ση2=0.05, ρ_ω=0.2, γ=0.9,
                    δ=0.01, β=0.95, ϵ=6.0, μν=1, σν2=0.09,
                    Smax=30, Ns=200, scale=1.0, size=100)

seed = 212311
burn_in = 100

firm_levels = [5000, 10000]
month_levels = [60, 240, 960]
labels = Dict(
    5000 => "low_firms",
    10000 => "medium_firms",
    60 => "low_months",
    240 => "med_months",
    960 => "high_months"
)

results = NamedTuple[]
for n_months in month_levels
    for n_firms in firm_levels
        label = "$(labels[n_firms])_$(labels[n_months])"
        println("Computing moments for $label (firms=$n_firms, months=$n_months)...")
        moments = summarize_target_moments(params;
                                           n_firms=n_firms,
                                           n_months=n_months,
                                           burn_in=burn_in,
                                           seed=seed)
        push!(results, (label=label, n_firms=n_firms, n_months=n_months, moments=moments))
    end
end

print_moment_table(results)
print_change_table(results)
