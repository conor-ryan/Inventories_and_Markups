using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

params_base = Parameters(c=1.0, fc=0.0, μη=log(0.05), ση2=0.05, ρ_ω=0.2, γ=0.9,
                         δ=0.01, β=0.95, ϵ=6.0, μν=1, σν2=0.09,
                         Ns=200, scale=1.0, size=100)

out_dir = joinpath(@__DIR__, "..", "SimulatedData")
target_moments_path = joinpath(out_dir, "target_moments.csv")
target_variances_path = joinpath(out_dir, "target_moment_variances.csv")
grid_path = joinpath(out_dir, "moments.csv")
results_path = joinpath(out_dir, "estimated_parameters.csv")
seed = 212311
sample_size = 1000
moment_names = (:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :μ_η)
moment_labels = String.(moment_names)

println("Loading saved target moments, moment variances, and grid...")
df_target_moments = CSV.read(target_moments_path, DataFrame)
df_target_variances = CSV.read(target_variances_path, DataFrame)
df_grid = CSV.read(grid_path, DataFrame)

df_target_moments.moment == moment_labels || error("target moments file has unexpected moment ordering")
df_target_variances.moment == moment_labels || error("target moment variances file has unexpected moment ordering")

target_moments = NamedTuple{Tuple(moment_names)}(Tuple(Float64.(df_target_moments.value)))
moment_variances = Float64.(df_target_variances.variance)
W = Diagonal(1.0 ./ moment_variances)

println("Selecting initial guess from precomputed grid...")
start_guess = select_best_grid_start(df_grid, target_moments, W)

println("Estimating parameters...")
ii_full = estimate_params_ii_full(params_base, target_moments, W;
                                  n_firms=5000,
                                  n_years=20,
                                  init_guess=[start_guess.γ, start_guess.μη, start_guess.ση2,
                                              start_guess.ρω, start_guess.σν2, start_guess.ϵ,
                                              start_guess.δ],
                                  max_iter=500,
                                  seed=seed,
                                  verbose=true,
                                  g_abstol=1e-1)

params_hat = Parameters(c=params_base.c, fc=params_base.fc,
                        μη=ii_full.μη, ση2=ii_full.ση2, ρ_ω=ii_full.ρω, γ=ii_full.γ̂,
                        δ=ii_full.δ̂, β=params_base.β, ϵ=ii_full.ϵ̂,
                        μν=params_base.μν, σν2=ii_full.σν2,
                        Ns=params_base.Ns,
                        scale=1.0, size=params_base.size)

println("Computing standard errors...")
se_results = compute_full_ii_asymptotic_variance(params_hat, W;
                                                 n_firms=5000,
                                                 n_years=20,
                                                 seed=seed,
                                                 solve_maxiter=1000,
                                                 sample_size=sample_size)

df_results = DataFrame(
    parameter = String.((:γ, :μη, :ση2, :ρω, :σν2, :ϵ, :δ)),
    estimate = [ii_full.γ̂, ii_full.μη, ii_full.ση2, ii_full.ρω, ii_full.σν2, ii_full.ϵ̂, ii_full.δ̂],
    std_error = se_results.se
)

CSV.write(results_path, df_results)
println("Saved estimates to $(results_path)")