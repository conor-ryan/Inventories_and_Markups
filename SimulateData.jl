using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV
include("ModelFunctions.jl")

"""
    simulate_panel_data(params, price_policy_interp, order_policy_interp;
                        N=1000, M=60, burn_in=100, seed=nothing)

Simulate a balanced panel of N firms over M months.

Each (firm, month) row contains:
- `firm_id`            : integer firm identifier (1..N)
- `month_id`           : integer month identifier (1..M)
- `sales`              : realized sales quantity D = min(ν p^{-ϵ}, s)
- `revenue`            : price × sales  (p·D)
- `cogs`               : procurement cost × sales  (c·D)
- `bom_inventory`      : beginning-of-month inventory s
- `operating_expenses` : ω·D^γ
- `inv_to_sales`       : bom_inventory / sales  (NaN when sales = 0)

A burn-in of `burn_in` periods is discarded per firm to draw from the
ergodic distribution rather than the arbitrary initial state.
"""
function simulate_panel_data(params::Parameters,
                              price_policy_interp,
                              order_policy_interp;
                              N::Int                   = 1000,
                              M::Int                   = 60,
                              burn_in::Int             = 100,
                              seed::Union{Int,Nothing} = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_obs      = N * M
    firm_ids   = Vector{Int}(undef, n_obs)
    month_ids  = Vector{Int}(undef, n_obs)
    sales_v    = Vector{Float64}(undef, n_obs)
    revenue_v  = Vector{Float64}(undef, n_obs)
    cogs_v     = Vector{Float64}(undef, n_obs)
    bom_inv_v  = Vector{Float64}(undef, n_obs)
    opex_v     = Vector{Float64}(undef, n_obs)
    isr_v      = Vector{Float64}(undef, n_obs)

    for firm in 1:N
        # Draw initial state from ergodic distributions
        s_current = rand(params.Sgrid)
        ω_idx     = draw_ω_index_ergodic(params)

        # Burn-in: advance the state without recording
        for _ in 1:burn_in
            ω_current = params.ω_grid[ω_idx]
            p_opt     = price_policy_interp(s_current, ω_current)
            n_opt     = order_policy_interp(s_current, ω_current)
            ν         = rand(params.dist)
            D         = min(ν * p_opt^(-params.ϵ), s_current)
            s_current = max((1 - params.δ) * (s_current - D + n_opt), 0.0)
            ω_idx     = draw_ω_index(params, ω_idx)
        end

        # Record M months
        row_start = (firm - 1) * M
        for month in 1:M
            ω_current = params.ω_grid[ω_idx]
            p_opt     = price_policy_interp(s_current, ω_current)
            n_opt     = order_policy_interp(s_current, ω_current)
            ν         = rand(params.dist)
            D         = min(ν * p_opt^(-params.ϵ), s_current)

            idx                = row_start + month
            firm_ids[idx]      = firm
            month_ids[idx]     = month
            sales_v[idx]       = D
            revenue_v[idx]     = p_opt * D
            cogs_v[idx]        = params.c * D
            bom_inv_v[idx]     = s_current
            opex_v[idx]        = operating_expense(ω_current, D, params)
            isr_v[idx]         = D > 0.0 ? s_current / (p_opt * D) : NaN

            s_current = max((1 - params.δ) * (s_current - D + n_opt), 0.0)
            ω_idx     = draw_ω_index(params, ω_idx)
        end
    end

    return DataFrame(
        firm_id            = firm_ids,
        month_id           = month_ids,
        sales              = sales_v,
        revenue            = revenue_v,
        cogs               = cogs_v,
        bom_inventory      = bom_inv_v,
        operating_expenses = opex_v,
        inv_to_sales       = isr_v
    )
end


# ---------------------------------------------------
# Main: solve model, simulate panel, save to CSV
# ---------------------------------------------------

params = Parameters(c=1.2, fc=0.0, μω=0.1, σω2=0.05, ρ_ω=0.1, γ=0.9,
                    δ=0.05, β=0.95, ϵ=6.0, μν=100, σν2=exp(7),
                    Smax=50, Ns=200, scale=1.0, size=3.0)

println("Solving model...")
p_policy, order_policy, V, V_by_omega, price_policy_interp, order_policy_interp, Vinterp =
    solve_model(params)
println("Model solved.")

println("Simulating panel data (N=1000, M=60, burn_in=100)...")
df = simulate_panel_data(params, price_policy_interp, order_policy_interp;
                          N=1000, M=60, burn_in=100, seed=212311)
println("Simulation complete. $(nrow(df)) observations.")
println(describe(df, :mean, :std, :min, :max))

out_dir  = joinpath(@__DIR__, "..", "SimulatedData")
mkpath(out_dir)
out_path = joinpath(out_dir, "simulated_panel.csv")
CSV.write(out_path, df)
println("Data written to $out_path")
