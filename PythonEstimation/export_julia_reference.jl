using DelimitedFiles

include("../ModelFunctions.jl")

output_dir = joinpath(@__DIR__, "parity_reference")
mkpath(output_dir)

Ns = 200
params = Parameters(
    c=1.0,
    fc=0.0,
    μη=log(0.01),
    ση2=0.05,
    ρ_ω=0.1,
    γ=0.9,
    δ=0.005,
    β=0.995,
    ϵ=16.0,
    μν=1.0,
    σν2=0.05,
    Ns=Ns,
    scale=1.0,
    size=100.0,
)

V, n_policy, p_policy, V_by_omega, converged = solve_value_function(
    params;
    tol=1e-3,
    maxiter=1000,
    full=false,
    fast_interp=true,
    conv=:policy,
    verbose=true,
)

writedlm(joinpath(output_dir, "sgrid.csv"), params.Sgrid, ',')
writedlm(joinpath(output_dir, "omega_grid.csv"), params.ω_grid, ',')
writedlm(joinpath(output_dir, "pi_omega.csv"), params.π_ω, ',')
writedlm(joinpath(output_dir, "p_omega.csv"), params.P_ω, ',')

writedlm(joinpath(output_dir, "v.csv"), V, ',')
writedlm(joinpath(output_dir, "n_policy.csv"), n_policy, ',')
writedlm(joinpath(output_dir, "p_policy.csv"), p_policy, ',')
writedlm(joinpath(output_dir, "v_by_omega.csv"), V_by_omega, ',')

writedlm(joinpath(output_dir, "converged.csv"), [converged ? 1 : 0], ',')

println("Saved Julia reference outputs to: ", output_dir)
