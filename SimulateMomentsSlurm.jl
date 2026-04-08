using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Plots, Interpolations, LineSearch, Random, Statistics, DataFrames, GLM, FixedEffectModels, Printf, CSV

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

# Baseline placeholders from SolveModel.jl (line 5).
params_base = Parameters(
    c=1.0,
    fc=0.0,
    μη=log(0.01),
    ση2=0.05,
    ρ_ω=0.1,
    γ=0.6,
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
ϵ_vals   = collect(range(4, 16, length=5))
σν2_vals = collect(range(0.25, 0.5, length=5))
δ_vals   = collect(range(0.025, 0.2, length=4))

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

function get_slurm_partition(n_total::Int)
    task_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
    task_min = parse(Int, get(ENV, "SLURM_ARRAY_TASK_MIN", string(task_id)))
    task_count = parse(Int, get(ENV, "SLURM_ARRAY_TASK_COUNT", "1"))
    rank = task_id - task_min + 1

    if task_count < 1
        error("SLURM_ARRAY_TASK_COUNT must be at least 1")
    end
    if rank < 1 || rank > task_count
        error("SLURM array rank $(rank) is out of bounds for count $(task_count)")
    end

    start_idx = fld((rank - 1) * n_total, task_count) + 1
    end_idx = fld(rank * n_total, task_count)
    return task_id, rank, task_count, start_idx, end_idx
end

function empty_results_df()
    return DataFrame(
        γ=Float64[],
        μη=Float64[],
        ση2=Float64[],
        ρω=Float64[],
        σν2=Float64[],
        ϵ=Float64[],
        δ=Float64[],
        avg_isr=Float64[],
        var_log1p_isr=Float64[],
        avg_gross_margin=Float64[],
        γ_OLS=Float64[],
        ρ_ω=Float64[],
        σ_η2=Float64[],
        μ_η=Float64[],
        failed=Bool[]
    )
end

mkpath("SimulatedData")

task_id, rank, task_count, start_idx, end_idx = get_slurm_partition(length(param_vectors))
output_path = "SimulatedData/moments_slurm_task_$(lpad(string(task_id), 3, '0')).csv"

println("Running SLURM parameter sweep task $(task_id) ($(rank) of $(task_count)).")
println("Assigned parameter indices: $(start_idx):$(end_idx) out of $(length(param_vectors)).")

if start_idx > end_idx
    println("No parameter vectors assigned to this task. Writing empty result file.")
    df_out = empty_results_df()
    CSV.write(output_path, df_out)
else
    param_vectors_local = param_vectors[start_idx:end_idx]
    println("Running parameter sweep with $(length(param_vectors_local)) local points...")

    df_out = compute_moments_on_grid(
        params_base,
        param_vectors_local;
        n_firms=40,
        n_years=10,
        seed=212311 + rank,
        output_path=output_path
    )
end

n_ok = sum(.!df_out.failed)
println("Sweep complete. $(n_ok) / $(nrow(df_out)) points succeeded.")
println("Saved: $(output_path)")

fail_fraction = nrow(df_out) == 0 ? NaN : sum(df_out.failed) / nrow(df_out)
if isnan(fail_fraction)
    println("Failure fraction: NaN (0 / 0)")
else
    @printf("Failure fraction: %.4f (%d / %d)\n", fail_fraction, sum(df_out.failed), nrow(df_out))
end

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
