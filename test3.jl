using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf
using BenchmarkTools, Profile

include("ModelFunctions.jl")
include("EstimationFunctions.jl")

params = Parameters(c=1.0, fc=0.0, μη=log(0.05), ση2=0.05, ρ_ω=0.2, γ=0.9,
                    δ=0.01, β=0.95, ϵ=6.0, μν=1, σν2=0.09,
                    Smax=30, Ns=200, scale=1.0, size=100)

n_firms = 5000
n_months = 240
n_years = div(n_months, 12)
seed = 212311
profile_repetitions = 20
solve_maxiter = 1000

println("Solving model once for benchmarking and profiling...")
_, _, _, _, ppi, opi, _, converged = solve_model(params; maxiter=solve_maxiter)
converged || error("solve_model did not converge")

println("Warming up _simulate_all_moments...")
_simulate_all_moments(params, ppi, opi, n_firms, n_years, seed)

println("\nBenchmarking _simulate_all_moments with @btime...")
benchmark = @benchmark _simulate_all_moments($params, $ppi, $opi, $n_firms, $n_years, $seed)
show(stdout, MIME("text/plain"), benchmark)
println()
println()
@btime _simulate_all_moments($params, $ppi, $opi, $n_firms, $n_years, $seed)

println("\nProfiling _simulate_all_moments with @Profile...")
Profile.clear()
@profile _simulate_all_moments(params, ppi, opi, n_firms, n_years, seed )

println("\nProfile summary:")
Profile.print(format=:tree, mincount=20, sortedby=:count)
