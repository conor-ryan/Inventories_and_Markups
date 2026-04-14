using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")
include("SimulateData.jl")

# ---------------------------------------------------
# Main: solve model, simulate panel, save to CSV
# ---------------------------------------------------

params = Parameters(c=1.0, fc=0.0, μη=log(0.05), ση2=0.05, ρ_ω=0.2, γ=0.9,
                    δ=0.01, β=0.95, ϵ=6.0, μν=1, σν2=0.09,
                    Ns=200, scale=1.0, size=100)

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

println("\n=== Bootstrap Moment Variances ===")
bootstrap_vars = bootstrap_moment_variances(df_monthly, df_annual;
                                            sample_fraction=0.5,
                                            n_boot=100,
                                            seed=212311)
@printf("avg_isr          = %10.6f\n", bootstrap_vars.variances.avg_isr)
@printf("var_log1p_isr    = %10.6f\n", bootstrap_vars.variances.var_log1p_isr)
@printf("avg_gross_margin = %10.6f\n", bootstrap_vars.variances.avg_gross_margin)
@printf("γ_OLS            = %10.6f\n", bootstrap_vars.variances.γ_OLS)
@printf("ρ_ω              = %10.6f\n", bootstrap_vars.variances.ρ_ω)
@printf("σ_η2             = %10.6f\n", bootstrap_vars.variances.σ_η2)
@printf("μ_η              = %10.6f\n", bootstrap_vars.variances.μ_η)

W = inv(bootstrap_vars.vcov)
# n = size(W,1)
# norm = ones(n)'*W*ones(n)
# W = W./norm

# ---------------------------------------------------
# Full 7-parameter indirect inference estimation
# ---------------------------------------------------

println("\n=== Monthly Data Moments ===")
target_moments = compute_full_ii_target_moments(df_monthly, df_annual)
@printf("avg_isr          = %10.6f\n", target_moments.avg_isr)
@printf("var_log1p_isr    = %10.6f\n", target_moments.var_log1p_isr)
@printf("avg_gross_margin = %10.6f\n", target_moments.avg_gross_margin)

println("\n=== Selecting Initial Guess from Precomputed Moments Grid ===")
df_grid = CSV.read(joinpath(out_dir, "moments.csv"), DataFrame)
start_guess = select_best_grid_start(df_grid, target_moments,W)
@printf("Best grid objective=%.6f | Start params: γ=%.6f, μη=%.6f, ση2=%.6f, ρω=%.6f, σν2=%.6f, ϵ=%.6f, δ=%.6f\n",
        start_guess.obj_value,
        start_guess.γ, start_guess.μη, start_guess.ση2, start_guess.ρω,
        start_guess.σν2, start_guess.ϵ, start_guess.δ)

println("\n=== Estimating all 7 parameters via Full Indirect Inference ===")
ii_full = estimate_params_ii_full(target_moments,
                                  [start_guess.γ, start_guess.μη, start_guess.ση2,
                                   start_guess.ρω, start_guess.σν2, start_guess.ϵ,
                                   start_guess.δ],
                                  W;
                                  n_firms  = 5000,
                                  n_years  = 20,
                                  max_iter = 500,
                                  seed     = 212311,
                                  verbose  = true,
                                  g_abstol=1e-1)

println("\n=== True vs Estimated (full 7-parameter estimator) ===")
println("Parameter   True          Estimated")
@printf("γ           %10.6f    %10.6f\n", params.γ, ii_full.γ̂)
@printf("μη          %10.6f    %10.6f\n", params.μη, ii_full.μη)
@printf("ση2         %10.6f    %10.6f\n", params.ση2, ii_full.ση2)
@printf("ρω          %10.6f    %10.6f\n", params.ρ_ω, ii_full.ρω)
@printf("σν2         %10.6f    %10.6f\n", params.σν2, ii_full.σν2)
@printf("ϵ           %10.6f    %10.6f\n", params.ϵ, ii_full.ϵ̂)
@printf("δ           %10.6f    %10.6f\n", params.δ, ii_full.δ̂)

params_hat = Parameters(c=params.c, fc=params.fc,
                        μη=ii_full.μη, ση2=ii_full.ση2, ρ_ω=ii_full.ρω, γ=ii_full.γ̂,
                        δ=ii_full.δ̂, β=params.β, ϵ=ii_full.ϵ̂,
                        μν=params.μν, σν2=ii_full.σν2,
                        Ns=params.Ns,
                        scale=1.0, size=params.size)

se_results = compute_full_ii_asymptotic_variance(params_hat, W;
                                                 n_firms=5000,
                                                 n_years=20,
                                                 seed=212311,
                                                 solve_maxiter=1000,
                                                 sample_size=1000)

println("\n=== Asymptotic Standard Errors ===")
println("Parameter   Estimate       Std. Error")
@printf("γ           %10.6f    %10.6f\n", ii_full.γ̂, se_results.se[1])
@printf("μη          %10.6f    %10.6f\n", ii_full.μη, se_results.se[2])
@printf("ση2         %10.6f    %10.6f\n", ii_full.ση2, se_results.se[3])
@printf("ρω          %10.6f    %10.6f\n", ii_full.ρω, se_results.se[4])
@printf("σν2         %10.6f    %10.6f\n", ii_full.σν2, se_results.se[5])
@printf("ϵ           %10.6f    %10.6f\n", ii_full.ϵ̂, se_results.se[6])
@printf("δ           %10.6f    %10.6f\n", ii_full.δ̂, se_results.se[7])
