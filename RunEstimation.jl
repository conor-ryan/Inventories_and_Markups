using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")
include("SimulateData.jl")

# ---------------------------------------------------
# Main: solve model, simulate panel, save to CSV
# ---------------------------------------------------

params = Parameters(c=1.0, fc=0.0, μη=log(0.1), ση2=0.05, ρ_ω=0.1, γ=0.9,
                    δ=0.05, β=0.95, ϵ=6.0, μν=1, σν2=0.09,
                    Smax=30, Ns=200, scale=1.0, size=100)

println("Solving model...")
println("Model solve will run inside simulate_panel_data.")

println("Simulating panel data (N=1000, M=60, burn_in=100)...")
df_monthly, df_annual = simulate_panel_data(params;
                                            N=1000, M=60, burn_in=100, seed=212311)
println("Simulation complete. $(nrow(df_monthly)) monthly observations, $(nrow(df_annual)) annual observations.")
println("\n--- Monthly summary ---")
println(describe(df_monthly, :mean, :std, :min, :max))
println("\n--- Annual summary ---")
println(describe(df_annual, :mean, :std, :min, :max))

out_dir = joinpath(@__DIR__, "..", "SimulatedData")
mkpath(out_dir)

monthly_path = joinpath(out_dir, "simulated_panel_monthly.csv")
annual_path  = joinpath(out_dir, "simulated_panel_annual.csv")
CSV.write(monthly_path, df_monthly)
CSV.write(annual_path, df_annual)
println("Monthly data written to $monthly_path")
println("Annual data written to  $annual_path")

# ---------------------------------------------------
# Full 7-parameter indirect inference estimation
# ---------------------------------------------------

println("\n=== Monthly Data Moments ===")
mo_moments = compute_monthly_moments(df_monthly)
@printf("avg_isr          = %10.6f\n", mo_moments.avg_isr)
@printf("var_log1p_isr    = %10.6f\n", mo_moments.var_log1p_isr)
@printf("avg_gross_margin = %10.6f\n", mo_moments.avg_gross_margin)

println("\n=== Estimating all 7 parameters via Full Indirect Inference ===")
ii_full = estimate_params_ii_full(params, df_monthly, df_annual;
                                  n_firms  = 100,
                                  n_years  = 25,
                                  max_iter = 500,
                                  seed     = 212311,
                                  verbose  = true)

println("\n=== True vs Estimated (full 7-parameter estimator) ===")
println("Parameter   True          Estimated")
@printf("γ           %10.6f    %10.6f\n", params.γ, ii_full.γ̂)
@printf("μη          %10.6f    %10.6f\n", params.μη, ii_full.μη)
@printf("ση2         %10.6f    %10.6f\n", params.ση2, ii_full.ση2)
@printf("ρω          %10.6f    %10.6f\n", params.ρ_ω, ii_full.ρω)
@printf("σν2         %10.6f    %10.6f\n", params.σν2, ii_full.σν2)
@printf("ϵ           %10.6f    %10.6f\n", params.ϵ, ii_full.ϵ̂)
@printf("δ           %10.6f    %10.6f\n", params.δ, ii_full.δ̂)
