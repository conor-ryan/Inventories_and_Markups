using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Plots, Interpolations, LineSearch, Random, Statistics, DataFrames, GLM, FixedEffectModels, Printf, CSV

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

# Baseline placeholders from SolveModel.jl (line 5).
params_base = Parameters(
    c=1.0,
    fc=0.0,
    μη=log(0.1),
    ση2=0.05,
    ρ_ω=0.1,
    γ=0.9,
    δ=0.05,
    β=0.95,
    ϵ=8.0,
    μν=1,
    σν2=0.09,
    Smax=20,
    Ns=200,
    scale=1.0,
    size=100
)

# Build ~100 points near the baseline by varying only (ϵ, σν2, δ).
ϵ_vals   = collect(range(params_base.ϵ - 1.0,  params_base.ϵ + 1.0,  length=5))
σν2_vals = collect(range(params_base.σν2 - 0.03, params_base.σν2 + 0.03, length=5))
δ_vals   = collect(range(max(0.001, params_base.δ - 0.02),
                         min(0.999, params_base.δ + 0.02),
                         length=4))

param_vectors = Vector{Vector{Float64}}()
sizehint!(param_vectors, length(ϵ_vals) * length(σν2_vals) * length(δ_vals))

for ϵ_i in ϵ_vals
    for σν2_i in σν2_vals
        for δ_i in δ_vals
            push!(param_vectors, [
                params_base.γ,
                params_base.μη,
                params_base.ση2,
                params_base.ρ_ω,
                σν2_i,
                ϵ_i,
                δ_i
            ])
        end
    end
end

println("Running parameter sweep with $(length(param_vectors)) points...")

df_out = compute_moments_on_grid(
    params_base,
    param_vectors;
    n_firms=40,
    n_years=10,
    seed=212311,
    output_path="SimulatedData/moments_example_100.csv"
)

n_ok = sum(.!df_out.failed)
println("Sweep complete. $(n_ok) / $(nrow(df_out)) points succeeded.")
println("Saved: SimulatedData/moments_example_100.csv")
