using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

out_dir = joinpath(@__DIR__, "..", "SimulatedData")
id_str = "001"

# Inputs for one simulated dataset and the shared precomputed moments grid.
target_moments_path = joinpath(out_dir, "target_moments_id_$(id_str).csv")
target_vcov_path = joinpath(out_dir, "target_moment_vcov_id_$(id_str).csv")
grid_path = joinpath(out_dir, "moments.csv")

# These should match the grid-generation settings used in compute_moments_on_grid.
seed_base = 212311
n_firms = 5000
n_years = 20
solve_maxiter = 500

moment_names = (:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :μ_η)
moment_labels = String.(moment_names)

println("Loading target moments and covariance for id $(id_str)...")
df_target_moments = CSV.read(target_moments_path, DataFrame)
df_target_vcov = CSV.read(target_vcov_path, DataFrame)
df_grid = CSV.read(grid_path, DataFrame)

if !(all(df_target_moments.moment .== moment_labels)) || !(all(df_target_vcov.moment .== moment_labels))
    error("Moment files have unexpected moment ordering")
end

target_moments = NamedTuple{Tuple(moment_names)}(Tuple(Float64.(df_target_moments.value)))
vcov = Matrix{Float64}(undef, length(moment_names), length(moment_names))
for (j, name) in enumerate(moment_names)
    vcov[:, j] = Float64.(df_target_vcov[!, String(name)])
end
W = inv(vcov)

println("Selecting best fit point from precomputed grid...")
start_guess = select_best_grid_start(df_grid, target_moments, W)
best_row = start_guess.row_index

@printf("Best row index on grid: %d\n", best_row)
@printf("Best grid objective: %.8f\n", start_guess.obj_value)
@printf("Best grid params: γ=%.6f μη=%.6f ση2=%.6f ρω=%.6f σν2=%.6f ϵ=%.6f δ=%.6f\n",
        start_guess.γ, start_guess.μη, start_guess.ση2, start_guess.ρω,
        start_guess.σν2, start_guess.ϵ, start_guess.δ)

# Re-solve model and re-simulate moments at best-grid parameters.
params_best = Parameters(μη=start_guess.μη, ση2=start_guess.ση2, ρ_ω=start_guess.ρω,
                         γ=start_guess.γ, δ=start_guess.δ, ϵ=start_guess.ϵ,
                         σν2=start_guess.σν2)

_, _, _, _, ppi, opi, _, converged = solve_model(params_best; maxiter=solve_maxiter,verbose=true)
converged || error("solve_model did not converge at best-grid parameters")

row_seed = seed_base + best_row - 1
moments_resim = _simulate_all_moments(params_best, ppi, opi, n_firms, n_years, row_seed)

moments_grid = [Float64(df_grid.avg_isr[best_row]),
                Float64(df_grid.var_log1p_isr[best_row]),
                Float64(df_grid.avg_gross_margin[best_row]),
                Float64(df_grid.γ_OLS[best_row]),
                Float64(df_grid.ρ_ω[best_row]),
                Float64(df_grid.σ_η2[best_row]),
                Float64(df_grid.μ_η[best_row])]

moments_recomputed = [moments_resim.avg_isr,
                      moments_resim.var_log1p_isr,
                      moments_resim.avg_gross_margin,
                      moments_resim.γ_OLS,
                      moments_resim.ρ_ω,
                      moments_resim.σ_η2,
                      moments_resim.μ_η]

target_vector = [target_moments.avg_isr,
                 target_moments.var_log1p_isr,
                 target_moments.avg_gross_margin,
                 target_moments.γ_OLS,
                 target_moments.ρ_ω,
                 target_moments.σ_η2,
                 target_moments.μ_η]

obj_grid = dot(target_vector - moments_grid, W * (target_vector - moments_grid))
obj_recomputed = dot(target_vector - moments_recomputed, W * (target_vector - moments_recomputed))

println("\nComparison of best-grid moments vs recomputed moments at same parameters:")
for i in eachindex(moment_names)
    @printf("  %-16s grid=%12.6f  recomputed=%12.6f  diff=%+12.6e\n",
            String(moment_names[i]), moments_grid[i], moments_recomputed[i], moments_recomputed[i] - moments_grid[i])
end

@printf("\nObjective at stored grid moments:      %.8f\n", obj_grid)
@printf("Objective at recomputed same params:   %.8f\n", obj_recomputed)
@printf("Objective difference (recomputed-grid): %+12.6e\n", obj_recomputed - obj_grid)
