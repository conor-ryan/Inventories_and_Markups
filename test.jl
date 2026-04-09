using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")
include("SimulateData.jl")

function full_ii_objective_from_moments(target_moments::NamedTuple, simulated_moments::NamedTuple)
    m_hat, w = _full_ii_mhat_weights(target_moments)
    m_tilde = [simulated_moments.avg_isr, simulated_moments.var_log1p_isr, simulated_moments.avg_gross_margin,
               simulated_moments.γ_OLS, simulated_moments.ρ_ω, simulated_moments.σ_η2, simulated_moments.μ_η]
    return sum(w[k] * (m_hat[k] - m_tilde[k])^2 for k in 1:7)
end

function row_moments(df_grid::DataFrame, idx::Int)
    return (
        avg_isr = Float64(df_grid.avg_isr[idx]),
        var_log1p_isr = Float64(df_grid.var_log1p_isr[idx]),
        avg_gross_margin = Float64(df_grid.avg_gross_margin[idx]),
        γ_OLS = Float64(df_grid.γ_OLS[idx]),
        ρ_ω = Float64(df_grid.ρ_ω[idx]),
        σ_η2 = Float64(df_grid.σ_η2[idx]),
        μ_η = Float64(df_grid.μ_η[idx])
    )
end

function params_from_start_guess(params_base::Parameters, start_guess::NamedTuple)
    return Parameters(
        c    = params_base.c,
        fc   = params_base.fc,
        μη   = start_guess.μη,
        ση2  = start_guess.ση2,
        ρ_ω  = start_guess.ρω,
        γ    = start_guess.γ,
        δ    = start_guess.δ,
        β    = params_base.β,
        ϵ    = start_guess.ϵ,
        μν   = params_base.μν,
        σν2  = start_guess.σν2,
        Smax = params_base.Smax,
        Ns   = params_base.Ns,
        size = params_base.size
    )
end

function print_moment_comparison(target_moments::NamedTuple,
                                 stored_moments::NamedTuple,
                                 fresh_moments::NamedTuple)
    println("\nmoment, target, stored_grid, fresh_simulation, stored_minus_fresh")
    rows = [
        ("avg_isr", target_moments.avg_isr, stored_moments.avg_isr, fresh_moments.avg_isr),
        ("var_log1p_isr", target_moments.var_log1p_isr, stored_moments.var_log1p_isr, fresh_moments.var_log1p_isr),
        ("avg_gross_margin", target_moments.avg_gross_margin, stored_moments.avg_gross_margin, fresh_moments.avg_gross_margin),
        ("γ_OLS", target_moments.γ_OLS, stored_moments.γ_OLS, fresh_moments.γ_OLS),
        ("ρ_ω", target_moments.ρ_ω, stored_moments.ρ_ω, fresh_moments.ρ_ω),
        ("σ_η2", target_moments.σ_η2, stored_moments.σ_η2, fresh_moments.σ_η2),
        ("μ_η", target_moments.μ_η, stored_moments.μ_η, fresh_moments.μ_η)
    ]

    for (name, target, stored, fresh) in rows
        @printf("%s, %.6f, %.6f, %.6f, %.6f\n", name, target, stored, fresh, stored - fresh)
    end
end

params = Parameters(c=1.0, fc=0.0, μη=log(0.05), ση2=0.05, ρ_ω=0.2, γ=0.9,
                    δ=0.01, β=0.95, ϵ=6.0, μν=1, σν2=0.09,
                    Smax=30, Ns=200, scale=1.0, size=100)

panel_n_firms = 1000
panel_n_months = 60
panel_burn_in = 100
panel_seed = 212311

est_n_firms = 100
est_n_years = 25
est_seed = 212311

grid_n_firms = 40
grid_n_years = 10
grid_seed = 212311

out_dir = joinpath(@__DIR__, "..", "SimulatedData")
grid_path = joinpath(out_dir, "moments.csv")
isfile(grid_path) || error("Missing grid file at $grid_path")

println("Simulating data used to construct target moments...")
df_monthly, df_annual = simulate_panel_data(params;
                                            N=panel_n_firms,
                                            M=panel_n_months,
                                            burn_in=panel_burn_in,
                                            seed=panel_seed)
target_moments = compute_full_ii_target_moments(df_monthly, df_annual)

println("Loading grid moments and selecting best point...")
df_grid = CSV.read(grid_path, DataFrame)
start_guess = select_best_grid_start(df_grid, target_moments)
stored_moments = row_moments(df_grid, start_guess.row_index)
stored_obj = full_ii_objective_from_moments(target_moments, stored_moments)

@printf("Best row index: %d\n", start_guess.row_index)
@printf("Grid objective at best row: %.6f\n", start_guess.obj_value)
@printf("Recomputed objective from stored row moments: %.6f\n", stored_obj)
@printf("Selected params: γ=%.6f, μη=%.6f, ση2=%.6f, ρω=%.6f, σν2=%.6f, ϵ=%.6f, δ=%.6f\n",
        start_guess.γ, start_guess.μη, start_guess.ση2, start_guess.ρω,
        start_guess.σν2, start_guess.ϵ, start_guess.δ)

println("Re-solving and re-simulating the selected parameter vector with estimation settings...")
params_start = params_from_start_guess(params, start_guess)
_, _, _, _, ppi, opi, _, converged = solve_model(params_start)
converged || error("solve_model did not converge at selected starting point")
fresh_moments = _simulate_all_moments(params_start, ppi, opi, est_n_firms, est_n_years, est_seed)
fresh_obj = full_ii_objective_from_moments(target_moments, fresh_moments)

println("Re-simulating the selected parameter vector with the original SimulateMoments settings...")
fresh_grid_moments = _simulate_all_moments(params_start, ppi, opi, grid_n_firms, grid_n_years, grid_seed)
fresh_grid_obj = full_ii_objective_from_moments(target_moments, fresh_grid_moments)

println("\nComparison against fresh simulation with estimation settings:")
print_moment_comparison(target_moments, stored_moments, fresh_moments)

println("\nComparison against fresh simulation with SimulateMoments settings:")
print_moment_comparison(target_moments, stored_moments, fresh_grid_moments)

println("\nobjective_summary, value")
@printf("stored_grid_objective, %.6f\n", stored_obj)
@printf("fresh_estimation_settings_objective, %.6f\n", fresh_obj)
@printf("fresh_simulate_moments_settings_objective, %.6f\n", fresh_grid_obj)
@printf("difference_estimation_minus_stored, %.6f\n", fresh_obj - stored_obj)
@printf("difference_simulate_moments_minus_stored, %.6f\n", fresh_grid_obj - stored_obj)
