using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf

include("ModelFunctions.jl")
include("EstimationFunctions.jl")
include("SimulationFunctions.jl")

# Parameter Bounds
ϵ_bounds   = (4.0, 20.0)
σν2_bounds = (0.01, 0.3)
δ_bounds   = (0.005, 0.1)
μη_bounds  = (log(0.0001), log(0.5))
γ_bounds   = (0.5, 1.25)
ση2_bounds = (0.025, 0.15)
ρ_bounds   = (0.0, 0.9)

num_data = 1

N = 1000
M = 60
burn_in = 100
seed = 212311
sample_fraction = 0.5
n_boot = 100

out_dir = joinpath(@__DIR__, "..", "SimulatedData")
mkpath(out_dir)

moment_names = [:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :μ_η]

draw_uniform(rng, bounds::Tuple{Float64,Float64}) = bounds[1] + rand(rng) * (bounds[2] - bounds[1])

rng = MersenneTwister(seed)
draws_df = DataFrame(
    id=Int[],
    γ=Float64[], μη=Float64[], ση2=Float64[], ρ_ω=Float64[], σν2=Float64[], ϵ=Float64[], δ=Float64[]
)

for id in 1:num_data
    γ_draw = draw_uniform(rng, γ_bounds)
    μη_draw = draw_uniform(rng, μη_bounds)
    ση2_draw = draw_uniform(rng, ση2_bounds)
    ρ_draw = draw_uniform(rng, ρ_bounds)
    σν2_draw = draw_uniform(rng, σν2_bounds)
    ϵ_draw = draw_uniform(rng, ϵ_bounds)
    δ_draw = draw_uniform(rng, δ_bounds)

    params = Parameters(c=1.0, fc=0.0, μη=μη_draw, ση2=ση2_draw, ρ_ω=ρ_draw, γ=γ_draw,
                        δ=δ_draw, β=0.95, ϵ=ϵ_draw, μν=1, σν2=σν2_draw,
                        Ns=200, scale=1.0, size=100)

    sim_seed = seed + id

    println("[", id, "/", num_data, "] Simulating panel data...")
    df_monthly, df_annual = simulate_panel_data(params;
                                                N=N,
                                                M=M,
                                                burn_in=burn_in,
                                                seed=sim_seed)

    println("[", id, "/", num_data, "] Computing target moments...")
    target_moments = compute_full_ii_target_moments(df_monthly, df_annual)

    println("[", id, "/", num_data, "] Computing bootstrap moment variances...")
    bootstrap_vars = bootstrap_moment_variances(df_monthly, df_annual;
                                                sample_fraction=sample_fraction,
                                                n_boot=n_boot,
                                                seed=sim_seed)

    id_str = lpad(string(id), 3, '0')
    moments_path = joinpath(out_dir, "target_moments_id_$(id_str).csv")
    vcov_path = joinpath(out_dir, "target_moment_vcov_id_$(id_str).csv")
    true_params_path = joinpath(out_dir, "true_parameters_id_$(id_str).csv")

    target_vector = [target_moments.avg_isr, target_moments.var_log1p_isr, target_moments.avg_gross_margin,
                     target_moments.γ_OLS, target_moments.ρ_ω, target_moments.σ_η2, target_moments.μ_η]
    df_mom = DataFrame(moment=String.(moment_names), value=target_vector)
    CSV.write(moments_path, df_mom)

    df_vcov = DataFrame(moment=[m for m in String.(moment_names)])
    for (j, name) in enumerate(moment_names)
        df_vcov[!, String(name)] = bootstrap_vars.vcov[:, j]
    end
    CSV.write(vcov_path, df_vcov)

    df_true = DataFrame(id=[id], γ=[γ_draw], μη=[μη_draw], ση2=[ση2_draw], ρ_ω=[ρ_draw],
                        σν2=[σν2_draw], ϵ=[ϵ_draw], δ=[δ_draw])
    CSV.write(true_params_path, df_true)

    push!(draws_df, (id=id, γ=γ_draw, μη=μη_draw, ση2=ση2_draw, ρ_ω=ρ_draw,
                     σν2=σν2_draw, ϵ=ϵ_draw, δ=δ_draw))

    println("[", id, "/", num_data, "] Saved target moments to ", moments_path)
    println("[", id, "/", num_data, "] Saved target moment covariance to ", vcov_path)
    println("[", id, "/", num_data, "] Saved true parameters to ", true_params_path)
end

draws_path = joinpath(out_dir, "simulated_parameter_draws.csv")
CSV.write(draws_path, draws_df)
println("Saved parameter draws to ", draws_path)