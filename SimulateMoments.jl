using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Plots, Interpolations, Random, Statistics, DataFrames, GLM, Printf, CSV

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

# Parameter Bounds
ϵ_bounds   = (4.0, 20.0)
σν2_bounds = (0.01, 0.3)
δ_bounds   = (0.01, 0.2)
μω_bounds  = (0.01, 0.2)
γ_bounds   = (0.8, 1.5)
ση2_bounds = (0.1, 1.0)
ρ_bounds   = (0.00, 0.9)

n_param_points = 250
param_bounds = [
    γ_bounds,
    μω_bounds,
    ση2_bounds,
    ρ_bounds,
    σν2_bounds,
    ϵ_bounds,
    δ_bounds
]

param_vectors = halton_param_vectors(param_bounds, n_param_points; seed=212311)

# Halton draws index 2 as μω = μη/(1-ρ); recover μη = μω * (1-ρ).
# Index 2 = μω, index 4 = ρ, index 7 = log(δ) (order: γ, μω→μη, ση2, ρ, σν2, ϵ, δ).
for v in param_vectors
    v[2] = log(v[2]) * (1.0 - v[4])
    # v[7] = exp(v[7])
end

# Slice param_vectors by SLURM array task ID (falls back to full set if not in a job array)
task_id    = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID",    "1"))
n_tasks    = parse(Int, get(ENV, "SLURM_ARRAY_TASK_COUNT", "1"))
chunk_size = ceil(Int, length(param_vectors) / n_tasks)
i_start    = (task_id - 1) * chunk_size + 1
i_end      = min(task_id * chunk_size, length(param_vectors))
param_vectors = param_vectors[i_start:i_end]

output_path = n_tasks > 1 ?
    "SimulatedData/moments_$(lpad(task_id, 3, '0')).csv" :
    "SimulatedData/moments.csv"

println("Task $(task_id)/$(n_tasks): grid points $(i_start)–$(i_end) ($(length(param_vectors)) total)")
# output_path = "../SimulatedData/moments.csv"

df_out = compute_moments_on_grid(
    param_vectors;
    n_firms=500,
    n_years=20,
    seed=212311,
    max_value_iterations=500,
    grid_size = 200, scale = 1.0, size = 100.0, solve_tol = 1e-2,
    output_path=output_path
)

n_ok = sum(.!df_out.failed)
println("Sweep complete. $(n_ok) / $(nrow(df_out)) points succeeded.")
println("Saved: $(output_path)")

# # Drop grid points with implausible inventory-to-sales ratios
# isr_threshold = 20.0
# n_before = nrow(df_out)
# df_out = df_out[coalesce.(df_out.avg_isr, Inf) .<= isr_threshold, :]
# n_dropped = n_before - nrow(df_out)
# n_dropped > 0 && @printf("Dropped %d / %d grid points with avg_isr > %.1f\n",
#                           n_dropped, n_before, isr_threshold)

# Summary output
fail_fraction = sum(df_out.failed) / nrow(df_out)
@printf("Failure fraction: %.4f (%d / %d)\n", fail_fraction, sum(df_out.failed), nrow(df_out))


df_success = df_out[(.!df_out.failed ).& (df_out.avg_isr.<4.0), :]
if nrow(df_success) == 0
    println("No successful simulations; moment summaries unavailable.")
else
    println("\nMoment summaries (successful simulations, $(nrow(df_success))):")
    println("moment, min,p10, p25, median, p75, p90,max")

    moment_cols = [:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :avg_opex_sales]
    for col in moment_cols
        vals = collect(skipmissing(df_success[!, col]))
        vals = vals[isfinite.(vals)]
        if isempty(vals)
            @printf("%s, NaN, NaN, NaN, NaN, NaN, NaN, NaN\n", String(col))
        else
            @printf("%s,%.6f, %.6f, %.6f, %.6f, %.6f,%.6f, %.6f\n",
                    String(col),
                    minimum(vals),
                    quantile(vals, 0.10),
                    quantile(vals, 0.25),
                    quantile(vals, 0.50),
                    quantile(vals, 0.75),
                    quantile(vals, 0.90),
                    maximum(vals))
        end
    end
end

# df_out[!,:μω] = exp.(df_out[!,:μη]./(1 .- df_out[!,:ρω]))

# df_success = df_out[.!df_out.failed, :]
# # df_success = df_success[df_success[!,:γ_OLS].>0,:]
# if nrow(df_success) == 0
#     println("No successful simulations; moment summaries unavailable.")
# else
#     println("\nParameter summaries (successful simulations only):")
#     println("parameter, min,p10, p25, median, p75, p90,max")

#     moment_cols = [:γ, :μω, :ση2, :ρω, :σν2, :ϵ, :δ]
#     for col in moment_cols
#         vals = collect(skipmissing(df_success[!, col]))
#         vals = vals[isfinite.(vals)]
#         if isempty(vals)
#             @printf("%s, NaN, NaN, NaN, NaN, NaN, NaN, NaN\n", String(col))
#         else
#             @printf("%s,%.6f, %.6f, %.6f, %.6f, %.6f,%.6f, %.6f\n",
#                     String(col),
#                     minimum(vals),
#                     quantile(vals, 0.10),
#                     quantile(vals, 0.25),
#                     quantile(vals, 0.50),
#                     quantile(vals, 0.75),
#                     quantile(vals, 0.90),
#                     maximum(vals))
#         end
#     end
# end