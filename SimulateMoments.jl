using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Plots, Interpolations, Random, Statistics, DataFrames, GLM, Printf, CSV

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

# Baseline placeholders from SolveModel.jl (line 5).
params = Parameters(c=1.0, fc=0.0, μη=log(0.01),ση2=0.05,ρ_ω=0.1, γ=0.9,δ=0.01, β=0.95, ϵ=8.0, μν=1, σν2=0.15, Smax=20, Ns=200,scale=1.0,size=100)



# Parameter Bounds
ϵ_bounds   = (4.0, 16.0)
σν2_bounds = (0.09, 0.21)
δ_bounds   = (0.005, 0.025)
μη_bounds  = (log(0.001), log(0.1))
γ_bounds   = (0.7, 1.1)
ση2_bounds = (0.025, 0.15)
ρ_bounds   = (0.0, 0.5)

n_param_points = 5000
param_bounds = [
    γ_bounds,
    μη_bounds,
    ση2_bounds,
    ρ_bounds,
    σν2_bounds,
    ϵ_bounds,
    δ_bounds
]

param_vectors = halton_param_vectors(param_bounds, n_param_points; seed=212311)

# param_vectors = param_vectors[1:100]

println("Running parameter sweep with $(length(param_vectors)) points...")

df_out = compute_moments_on_grid(
    params,
    param_vectors;
    n_firms=40,
    n_years=10,
    seed=212311,
    output_path="SimulatedData/moments.csv"
)

n_ok = sum(.!df_out.failed)
println("Sweep complete. $(n_ok) / $(nrow(df_out)) points succeeded.")
println("Saved: SimulatedData/moments.csv")

# Summary output
fail_fraction = sum(df_out.failed) / nrow(df_out)
@printf("Failure fraction: %.4f (%d / %d)\n", fail_fraction, sum(df_out.failed), nrow(df_out))

df_success = df_out[.!df_out.failed, :]
if nrow(df_success) == 0
    println("No successful simulations; moment summaries unavailable.")
else
    println("\nMoment summaries (successful simulations only):")
    println("moment, p25, median, mean, p75")

    moment_cols = [:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :μ_η]
    for col in moment_cols
        vals = collect(skipmissing(df_success[!, col]))
        vals = vals[isfinite.(vals)]
        if isempty(vals)
            @printf("%s, NaN, NaN, NaN, NaN\n", String(col))
        else
            @printf("%s, %.6f, %.6f, %.6f, %.6f\n",
                    String(col),
                    quantile(vals, 0.25),
                    quantile(vals, 0.50),
                    mean(vals),
                    quantile(vals, 0.75))
        end
    end
end
