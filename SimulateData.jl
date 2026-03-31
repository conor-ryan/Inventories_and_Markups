using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames, CSV, GLM, FixedEffectModels, Printf
include("ModelFunctions.jl")
include("EstimationFunctions.jl")

"""
    simulate_panel_data(params, price_policy_interp, order_policy_interp;
                        N=1000, M=60, burn_in=100, seed=nothing)

Simulate a balanced panel of N firms over M months. M must be a multiple of 12.

Returns `(monthly_df, annual_df)`.

Monthly DataFrame columns (N×M rows):
- `firm_id`            : integer firm identifier (1..N)
- `month_id`           : integer month identifier (1..M)
- `sales`              : realized sales quantity D = min(ν p^{-ϵ}, s)
- `revenue`            : price × sales  (p·D)
- `cogs`               : procurement cost × sales  (c·D)
- `bom_inventory`      : beginning-of-month inventory s
- `operating_expenses` : ω·D^γ
- `inv_to_sales`       : bom_inventory / revenue  (NaN when revenue = 0)

Annual DataFrame columns (N×(M÷12) rows):
- `firm_id`            : integer firm identifier
- `year_id`            : integer year identifier (1..M÷12)
- `total_sales`        : sum of monthly sales quantities
- `total_revenue`      : sum of monthly revenues
- `total_cogs`         : sum of monthly COGS
- `total_opex`         : sum of monthly operating expenses
- `boy_inventory`      : beginning-of-year inventory (BOM of first month of year)
- `eoy_inventory`      : end-of-year inventory (BOM of first month of following year)
- `inv_to_sales`       : boy_inventory / (total_revenue / 12)

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
    @assert M % 12 == 0 "M must be a multiple of 12 for annual aggregation"

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
    eom_inv_v  = Vector{Float64}(undef, n_obs)   # end-of-month inventory (internal use)
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
            s_next    = max((1 - params.δ) * (s_current - D + n_opt), 0.0)

            idx                = row_start + month
            firm_ids[idx]      = firm
            month_ids[idx]     = month
            sales_v[idx]       = D
            revenue_v[idx]     = p_opt * D
            cogs_v[idx]        = params.c * D
            bom_inv_v[idx]     = s_current
            eom_inv_v[idx]     = s_next
            opex_v[idx]        = operating_expense(ω_current, D, params)
            rev                = p_opt * D
            isr_v[idx]         = rev > 0.0 ? s_current / rev : NaN

            s_current = s_next
            ω_idx     = draw_ω_index(params, ω_idx)
        end
    end

    monthly_df = DataFrame(
        firm_id            = firm_ids,
        month_id           = month_ids,
        sales              = sales_v,
        revenue            = revenue_v,
        cogs               = cogs_v,
        bom_inventory      = bom_inv_v,
        operating_expenses = opex_v,
        inv_to_sales       = isr_v
    )

    # ---------------------------------------------------
    # Build annual panel
    # ---------------------------------------------------
    n_years       = M ÷ 12
    n_ann         = N * n_years
    ann_firm_ids  = Vector{Int}(undef, n_ann)
    ann_year_ids  = Vector{Int}(undef, n_ann)
    ann_sales     = Vector{Float64}(undef, n_ann)
    ann_revenue   = Vector{Float64}(undef, n_ann)
    ann_cogs      = Vector{Float64}(undef, n_ann)
    ann_opex      = Vector{Float64}(undef, n_ann)
    ann_boy_inv   = Vector{Float64}(undef, n_ann)
    ann_eoy_inv   = Vector{Float64}(undef, n_ann)
    ann_isr       = Vector{Float64}(undef, n_ann)

    for firm in 1:N
        row_start_m = (firm - 1) * M        # offset into monthly vectors
        row_start_a = (firm - 1) * n_years  # offset into annual vectors
        for yr in 1:n_years
            m_first = row_start_m + (yr - 1) * 12 + 1
            m_last  = row_start_m + yr * 12
            a_idx   = row_start_a + yr

            tot_sales   = sum(sales_v[m_first:m_last])
            tot_revenue = sum(revenue_v[m_first:m_last])
            tot_cogs    = sum(cogs_v[m_first:m_last])
            tot_opex    = sum(opex_v[m_first:m_last])
            boy         = bom_inv_v[m_first]
            eoy         = eom_inv_v[m_last]
            avg_monthly_rev = tot_revenue / 12

            ann_firm_ids[a_idx]  = firm
            ann_year_ids[a_idx]  = yr
            ann_sales[a_idx]     = tot_sales
            ann_revenue[a_idx]   = tot_revenue
            ann_cogs[a_idx]      = tot_cogs
            ann_opex[a_idx]      = tot_opex
            ann_boy_inv[a_idx]   = boy
            ann_eoy_inv[a_idx]   = eoy
            ann_isr[a_idx]       = avg_monthly_rev > 0.0 ? boy / avg_monthly_rev : NaN
        end
    end

    annual_df = DataFrame(
        firm_id       = ann_firm_ids,
        year_id       = ann_year_ids,
        total_sales   = ann_sales,
        total_revenue = ann_revenue,
        total_cogs    = ann_cogs,
        total_opex    = ann_opex,
        boy_inventory = ann_boy_inv,
        eoy_inventory = ann_eoy_inv,
        inv_to_sales  = ann_isr
    )

    return monthly_df, annual_df
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
df_monthly, df_annual = simulate_panel_data(params, price_policy_interp, order_policy_interp;
                                             N=1000, M=60, burn_in=100, seed=212311)
println("Simulation complete. $(nrow(df_monthly)) monthly observations, $(nrow(df_annual)) annual observations.")
println("\n--- Monthly summary ---")
println(describe(df_monthly, :mean, :std, :min, :max))
println("\n--- Annual summary ---")
println(describe(df_annual, :mean, :std, :min, :max))

out_dir  = joinpath(@__DIR__, "..", "SimulatedData")
mkpath(out_dir)

monthly_path = joinpath(out_dir, "simulated_panel_monthly.csv")
annual_path  = joinpath(out_dir, "simulated_panel_annual.csv")
CSV.write(monthly_path, df_monthly)
CSV.write(annual_path,  df_annual)
println("Monthly data written to $monthly_path")
println("Annual data written to  $annual_path")


# ---------------------------------------------------
# Estimate γ from the annual simulated data using
# the iterative bias-corrected IV procedure
# ---------------------------------------------------

println("\n=== Auxiliary OLS Regression on Annual Data ===")
ψ̂_data = compute_annual_auxiliary(df_annual)
display(coeftable(ψ̂_data.ols_result))

println("\n=== Estimating γ, μω, σω2, ρω from Annual Data via Indirect Inference ===")
ii_result = estimate_params_ii_annual(params, df_annual;
                                       n_firms   = 200,
                                       n_years   = 50,
                                       max_iter  = 500,
                                       seed      = 212311,
                                       verbose   = true)

println("\n=== True vs Estimated ===")
println("Parameter   True          Estimated (monthly)")
@printf("γ           %10.6f    %10.6f\n", params.γ,   ii_result.γ̂)
@printf("μω          %10.6f    %10.6f\n", exp(params.μω), ii_result.μω_monthly)
@printf("σω2         %10.6f    %10.6f\n", params.σω2,  ii_result.σω2_monthly)
@printf("ρω          %10.6f    %10.6f\n", params.ρ_ω,  ii_result.ρω_monthly)
