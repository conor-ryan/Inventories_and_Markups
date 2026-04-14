using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")


out_dir = joinpath(@__DIR__, "..", "SimulatedData")
target_moments_path = joinpath(out_dir, "target_moments.csv")
target_vcov_path = joinpath(out_dir, "target_moment_vcov.csv")
grid_path = joinpath(out_dir, "moments.csv")
results_path = joinpath(out_dir, "estimated_parameters.csv")
seed = 212311
sample_size = 1000
moment_names = (:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :μ_η)
moment_labels = String.(moment_names)

println("Loading saved target moments, moment covariance matrix, and grid...")
df_target_moments = CSV.read(target_moments_path, DataFrame)
df_target_vcov = CSV.read(target_vcov_path, DataFrame)
df_grid = CSV.read(grid_path, DataFrame)

if !(all(df_target_moments.moment .== moment_labels)) | !(all(df_target_vcov.moment .== moment_labels ))
    error("target moments file has unexpected moment ordering")
end

target_moments = NamedTuple{Tuple(moment_names)}(Tuple(Float64.(df_target_moments.value)))
vcov = Matrix{Float64}(undef, length(moment_names), length(moment_names))
for (j, name) in enumerate(moment_names)
    vcov[:, j] = Float64.(df_target_vcov[!, String(name)])
end
W = inv(vcov)

println("Selecting initial guess from precomputed grid...")
start_guess = select_best_grid_start(df_grid, target_moments, W)
@printf("Best pre-computed grid objective value: %.6f\n", start_guess.obj_value)

best_row = start_guess.row_index
println("Moments at best pre-computed grid point:")
@printf("  avg_isr          = %10.6f\n", Float64(df_grid.avg_isr[best_row]))
@printf("  var_log1p_isr    = %10.6f\n", Float64(df_grid.var_log1p_isr[best_row]))
@printf("  avg_gross_margin = %10.6f\n", Float64(df_grid.avg_gross_margin[best_row]))
@printf("  γ_OLS            = %10.6f\n", Float64(df_grid.γ_OLS[best_row]))
@printf("  ρ_ω              = %10.6f\n", Float64(df_grid.ρ_ω[best_row]))
@printf("  σ_η2             = %10.6f\n", Float64(df_grid.σ_η2[best_row]))
@printf("  μ_η              = %10.6f\n", Float64(df_grid.μ_η[best_row]))

println("Estimating parameters...")
ii_full = estimate_params_ii_full(target_moments,
                                  [start_guess.γ, start_guess.μη, start_guess.ση2,
                                   start_guess.ρω, start_guess.σν2, start_guess.ϵ,
                                   start_guess.δ],
                                  W;
                                  n_firms=5000,
                                  n_years=20,
                                  max_iter=500,
                                  seed=seed,
                                  verbose=true,
                                  g_abstol=1e-1)

params_hat = Parameters(μη=ii_full.μη, ση2=ii_full.ση2, ρ_ω=ii_full.ρω, γ=ii_full.γ,
                        δ=ii_full.δ, ϵ=ii_full.ϵ,
                        σν2=ii_full.σν2)

println("Computing standard errors...")
se_results = compute_full_ii_asymptotic_variance(params_hat, W;
                                                 n_firms=5000,
                                                 n_years=20,
                                                 seed=seed,
                                                 solve_maxiter=1000,
                                                 sample_size=sample_size)

df_results = DataFrame(
    parameter = String.((:γ, :μη, :ση2, :ρω, :σν2, :ϵ, :δ)),
    estimate = [ii_full.γ, ii_full.μη, ii_full.ση2, ii_full.ρω, ii_full.σν2, ii_full.ϵ, ii_full.δ],
    std_error = se_results.se
)

CSV.write(results_path, df_results)
println("Saved estimates to $(results_path)")