using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations,
      Random, Statistics, DataFrames
"""
    simulate_panel_data(params;
                        N=1000, M=60, burn_in=100, seed=nothing,
                        solve_verbose=false, solve_maxiter=1000)

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
function simulate_panel_data(params::Parameters;
                              N::Int                   = 1000,
                              M::Int                   = 60,
                              burn_in::Int             = 100,
                              seed::Union{Int,Nothing} = nothing,
                              solve_verbose::Bool      = false,
                              solve_maxiter::Int       = 1000)
    @assert M % 12 == 0 "M must be a multiple of 12 for annual aggregation"

    _, _, _, _, price_policy_interp, order_policy_interp, _, vf_converged =
        solve_model(params, verbose=solve_verbose, maxiter=solve_maxiter)
    vf_converged || error("solve_model did not converge within solve_maxiter=$(solve_maxiter)")

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


"""
    bootstrap_sample_panel(df_monthly, df_annual; sample_fraction=0.5, rng=Random.default_rng())

Draw a firm-level bootstrap sample from a simulated monthly/annual panel.

Firms are sampled with replacement, and each sampled firm is assigned a new
bootstrap firm id so repeated draws of the same firm remain separate panel
units in the resampled data.

Returns `(monthly_boot, annual_boot)`.
"""
function bootstrap_sample_panel(df_monthly::DataFrame,
                                df_annual::DataFrame;
                                sample_fraction::Float64 = 0.5,
                                rng::AbstractRNG = Random.default_rng())
    0.0 < sample_fraction <= 1.0 || error("sample_fraction must lie in (0, 1]")

    firm_ids = unique(df_monthly.firm_id)
    n_firms = length(firm_ids)
    n_draws = max(1, round(Int, sample_fraction * n_firms))
    sampled_firms = rand(rng, firm_ids, n_draws)

    monthly_parts = DataFrame[]
    annual_parts = DataFrame[]

    for (boot_firm_id, source_firm_id) in enumerate(sampled_firms)
        monthly_part = copy(df_monthly[df_monthly.firm_id .== source_firm_id, :])
        annual_part = copy(df_annual[df_annual.firm_id .== source_firm_id, :])
        monthly_part.firm_id .= boot_firm_id
        annual_part.firm_id .= boot_firm_id
        push!(monthly_parts, monthly_part)
        push!(annual_parts, annual_part)
    end

    return vcat(monthly_parts...), vcat(annual_parts...)
end


"""
    bootstrap_moments(df_monthly, df_annual; sample_fraction=0.5, n_boot=100, seed=nothing)

Compute the full indirect-inference moments on repeated firm-level bootstrap
samples of the supplied monthly and annual panels.

Returns a vector of NamedTuples, one per bootstrap draw, each with fields:
`avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, μ_η`.

Requires `compute_full_ii_target_moments` to be available in scope.
"""
function bootstrap_moments(df_monthly::DataFrame,
                           df_annual::DataFrame;
                           sample_fraction::Float64 = 0.5,
                           n_boot::Int = 100,
                           seed::Union{Int,Nothing} = nothing)
    n_boot > 0 || error("n_boot must be positive")
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    draws = Vector{NamedTuple}(undef, n_boot)
    for b in 1:n_boot
        monthly_boot, annual_boot = bootstrap_sample_panel(df_monthly, df_annual;
                                                           sample_fraction=sample_fraction,
                                                           rng=rng)
        draws[b] = compute_full_ii_target_moments(monthly_boot, annual_boot)
    end

    return draws
end


"""
    bootstrap_moment_variances(df_monthly, df_annual; sample_fraction=0.5, n_boot=100, seed=nothing)

Run the firm-level bootstrap and return the variance of each of the seven
moments across bootstrap draws, along with their variance-covariance matrix.

Returns a NamedTuple with fields:
`variances`, `vcov`, `moment_names`.

- `variances` is a NamedTuple with fields
    `avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, μ_η`
- `vcov` is the full variance-covariance matrix in the same moment order
- `moment_names` records that order explicitly
"""
function bootstrap_moment_variances(df_monthly::DataFrame,
                                    df_annual::DataFrame;
                                    sample_fraction::Float64 = 0.5,
                                    n_boot::Int = 100,
                                    seed::Union{Int,Nothing} = nothing)
    draws = bootstrap_moments(df_monthly, df_annual;
                              sample_fraction=sample_fraction,
                              n_boot=n_boot,
                              seed=seed)

    moment_names = (:avg_isr, :var_log1p_isr, :avg_gross_margin, :γ_OLS, :ρ_ω, :σ_η2, :μ_η)
    moment_matrix = hcat([
        [draw.avg_isr, draw.var_log1p_isr, draw.avg_gross_margin,
         draw.γ_OLS, draw.ρ_ω, draw.σ_η2, draw.μ_η]
        for draw in draws
    ]...)'
    vcov = cov(moment_matrix)

    variances = (
        avg_isr = vcov[1, 1],
        var_log1p_isr = vcov[2, 2],
        avg_gross_margin = vcov[3, 3],
        γ_OLS = vcov[4, 4],
        ρ_ω = vcov[5, 5],
        σ_η2 = vcov[6, 6],
        μ_η = vcov[7, 7]
    )

    return (
        variances = variances,
        vcov = vcov,
        moment_names = moment_names
    )
end


