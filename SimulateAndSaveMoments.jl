using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")
include("SimulationFunctions.jl")

params = Parameters(c=1.0, fc=0.0, μη=log(0.05), ση2=0.05, ρ_ω=0.2, γ=0.9,
                    δ=0.01, β=0.95, ϵ=6.0, μν=1, σν2=0.09,
                    Ns=200, scale=1.0, size=100)

N = 1000
M = 60
burn_in = 100
seed = 212311
sample_fraction = 0.5
n_boot = 100

out_dir = joinpath(@__DIR__, "..", "SimulatedData")
mkpath(out_dir)

println("Simulating panel data...")
df_monthly, df_annual = simulate_panel_data(params;
                                            N=N,
                                            M=M,
                                            burn_in=burn_in,
                                            seed=seed)

println("Computing target moments...")
target_moments = compute_full_ii_target_moments(df_monthly, df_annual)

println("Computing bootstrap moment variances...")
bootstrap_vars = bootstrap_moment_variances(df_monthly, df_annual;
                                            sample_fraction=sample_fraction,
                                            n_boot=n_boot,
                                            seed=seed)

saved_paths = save_full_ii_moment_inputs(target_moments, bootstrap_vars;
                                         output_dir=out_dir)

println("Saved target moments to $(saved_paths.moments_path)")
println("Saved target moment variances to $(saved_paths.variances_path)")
println("Saved target moment covariance to $(saved_paths.vcov_path)")