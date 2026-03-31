"""
    estimate_gamma_bc(params, df, n_periods; n_firms, max_iter, tol, seed)

Estimate the cost-function curvature parameter γ via IV (using Δ(inv/sales) as
instrument) followed by iterative model-based bias correction.

# Arguments
- `params`    : Parameters object used as template for all fixed model parameters
- `df`        : DataFrame with columns `log_expense`, `log_demand`, `Δinv_sales`
- `n_periods` : number of simulated periods per firm

# Returns
`(γ̂_BC, μω, σω2, ρω)` — bias-corrected γ and estimated ω process parameters
"""
function estimate_gamma_bc(params::Parameters, df::DataFrame;
                            n_periods::Int = 25000,
                            n_firms::Int  = 40,
                            max_iter::Int = 20,
                            tol::Real     = 1e-2,
                            seed::Int     = 212311)

    # --- Step 1: initial IV estimate ---
    iv      = reg(df, @formula(log_expense ~ (log_demand ~ Δinv_sales)))
    γ̂_step1 = coef(iv)[end]

    log_ω_proxy = coef(iv)[1] .+ FixedEffectModels.residuals(iv, df)
    μω_current, σω2_current, ρω_current = estimate_omega_ar1(log_ω_proxy, df.firm_boundary)

    println("\n=== Iterative Bias-Corrected Estimation ===")
    println("Step 1 — Initial γ̂ (z-IV):  $(round(γ̂_step1, digits=6))")
    println("\n Iter │     γ̂        │    bias     │   γ̂_BC      │   μ̂_ω      │   σ̂η²      │   ρ̂_ω")
    println("──────┼──────────────┼─────────────┼─────────────┼─────────────┼─────────────┼───────────")

    γ̂_current = γ̂_step1
    γ̂_BC      = γ̂_current

    for iter in 1:max_iter
        # Re-solve and re-simulate at current (γ, μω, σω2, ρω)
        μ_ν_level  = exp(params.μν + 0.5 * params.σν2)
        σ_ν2_level = (exp(params.σν2) - 1.0) * μ_ν_level^2
        params_iter = Parameters(c=params.c, fc=params.fc, μω=μω_current, σω2=σω2_current,
                                  ρ_ω=ρω_current, γ=γ̂_current,
                                  δ=params.δ, β=params.β, ϵ=params.ϵ,
                                  μν=μ_ν_level, σν2=σ_ν2_level,
                                  Smax=params.Smax, Ns=params.Ns)
        _, _, _, _, ppi_iter, opi_iter, _ = solve_model(params_iter)
        Random.seed!(seed)
        _, _, dem_i, _, exp_i, ω_i, isr_i =
            simulate_firm(n_firms, n_periods, ppi_iter, opi_iter, params_iter)

        # Model-implied bias: plim(γ̂_z-IV) = γ + Cov(z, log ω) / Cov(z, log D)
        mask_i     = (exp_i .> 0) .& (dem_i .> 0) .& (ω_i .> 0)
        Δisr_i     = similar(isr_i)
        Δisr_i[1]  = NaN
        for t in 2:length(isr_i)
            Δisr_i[t] = (t - 1) % n_periods == 0 ? NaN : isr_i[t] - isr_i[t - 1]
        end
        valid_i = mask_i .& .!isnan.(Δisr_i)
        bias_i  = cov(Δisr_i[valid_i], log.(ω_i[valid_i])) /
                  cov(Δisr_i[valid_i], log.(dem_i[valid_i]))

        # Re-estimate ω from original data using current γ̂
        log_ω_hat = df.log_expense .- γ̂_current .* df.log_demand
        μ̂_ω_new, σ̂η2_new, ρ̂_ω_new = estimate_omega_ar1(log_ω_hat, df.firm_boundary)

        # Bias-corrected γ
        γ̂_BC_new = γ̂_step1 - bias_i

        @printf("  %3d  │  %10.6f  │  %10.6f │  %10.6f │  %10.6f │  %10.6f │  %10.6f\n",
                iter, γ̂_current, bias_i, γ̂_BC_new, μ̂_ω_new, σ̂η2_new, ρ̂_ω_new)

        converged   = abs(γ̂_BC_new - γ̂_BC) < tol
        γ̂_BC        = γ̂_BC_new
        γ̂_current   = γ̂_BC_new
        μω_current  = μ̂_ω_new
        σω2_current = σ̂η2_new
        ρω_current  = ρ̂_ω_new

        if converged
            println("Converged at iteration $iter.")
            break
        end
    end

    println("\nFinal bias-corrected γ̂^BC: $(round(γ̂_BC, digits=6))")
    println("Final ω estimates  —  μω: $(round(μω_current, digits=6))  σω2: $(round(σω2_current, digits=6))  ρω: $(round(ρω_current, digits=6))")

    return γ̂_BC, μω_current, σω2_current, ρω_current
end


"""
    estimate_omega_ar1(log_ω_proxy, firm_boundary)

Fit an AR(1) to a panel of log(ω) proxies and return the level mean, innovation
variance, and persistence.  `log_ω_proxy` is a vector with observations stacked
across firms.  `firm_boundary` is a `Bool` vector of the same length whose `true`
entries mark the first observation of each firm (where no AR(1) lag exists).

Returns `(μω, σω2, ρω)` where
- `μω`  = exp(unconditional mean of log ω)
- `σω2` = variance of the AR(1) innovation
- `ρω`  = AR(1) slope coefficient
"""
function estimate_omega_ar1(log_ω_proxy::AbstractVector{<:Real}, firm_boundary::AbstractVector{Bool})
    n   = length(log_ω_proxy)
    lag = fill(NaN, n)
    for t in 2:n
        lag[t] = firm_boundary[t] ? NaN : log_ω_proxy[t - 1]
    end
    keep = .!isnan.(lag)
    y    = log_ω_proxy[keep]
    x    = lag[keep]
    # OLS: y = a + ρ·x
    x̄, ȳ  = mean(x), mean(y)
    ρω    = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
    a     = ȳ - ρω * x̄
    resid = y .- (a .+ ρω .* x)
    μω    = exp(a / (1 - ρω))   # unconditional mean level
    σω2   = var(resid)          # innovation variance
    return μω, σω2, ρω
end


"""
    annual_ar1_to_monthly(μω, σω2_innov_annual, ρω_annual)

Convert annual-frequency AR(1) parameters to monthly-frequency AR(1) parameters.

Mapping:
- ρ_monthly  = ρ_annual^(1/12)
- μω is frequency-independent (unconditional mean of ω in levels)
- σω2_monthly is chosen to preserve the unconditional variance of log(ω):
    σω2_innov / (1 - ρ²) is held equal across frequencies

Returns `(μω, σω2_monthly, ρω_monthly)`.
"""
function annual_ar1_to_monthly(μω::Float64, σω2_innov_annual::Float64, ρω_annual::Float64)
    ρω_monthly  = sign(ρω_annual) * abs(ρω_annual)^(1/12)
    denom       = max(1 - ρω_annual^2, 1e-10)
    σω2_monthly = σω2_innov_annual * (1 - ρω_monthly^2) / denom
    return μω, σω2_monthly, ρω_monthly
end


"""
    compute_annual_sim_bias(exp_i, dem_i, ω_i, isr_i, n_firms, n_months)

From the monthly simulation outputs of `simulate_firm`, aggregate to annual
frequency and return the model-implied IV bias at the annual level:

    bias = Cov(Δz_annual, log ω_annual) / Cov(Δz_annual, log D_annual)

where:
- z_annual  = monthly ISR at the beginning of the year (from `isr_i`)
- ω_annual  = geometric mean of monthly ω over the year
- D_annual  = total annual sales quantity
"""
function compute_annual_sim_bias(exp_i, dem_i, ω_i, isr_i,
                                  n_firms::Int, n_months::Int)
    n_years   = n_months ÷ 12
    n_obs_ann = n_firms * n_years

    ann_log_demand = Vector{Float64}(undef, n_obs_ann)
    ann_log_ω      = Vector{Float64}(undef, n_obs_ann)
    ann_isr        = Vector{Float64}(undef, n_obs_ann)

    for firm in 1:n_firms
        for yr in 1:n_years
            m_first = (firm - 1) * n_months + (yr - 1) * 12 + 1
            m_last  = m_first + 11
            a_idx   = (firm - 1) * n_years + yr

            tot_demand         = sum(dem_i[m_first:m_last])
            geom_ω             = exp(mean(log.(max.(ω_i[m_first:m_last], eps()))))
            ann_log_demand[a_idx] = tot_demand > 0 ? log(tot_demand) : NaN
            ann_log_ω[a_idx]      = log(geom_ω)
            ann_isr[a_idx]        = isr_i[m_first]   # monthly ISR at BOY
        end
    end

    # Year-over-year Δ within each firm
    Δann_isr = fill(NaN, n_obs_ann)
    for firm in 1:n_firms
        for yr in 2:n_years
            idx          = (firm - 1) * n_years + yr
            Δann_isr[idx] = ann_isr[idx] - ann_isr[idx - 1]
        end
    end

    valid = .!isnan.(Δann_isr) .& isfinite.(ann_log_ω) .& isfinite.(ann_log_demand)
    return cov(Δann_isr[valid], ann_log_ω[valid]) /
           cov(Δann_isr[valid], ann_log_demand[valid])
end


"""
    estimate_gamma_bc_annual(params, df_annual; n_years, n_firms, max_iter, tol, seed)

Annual-frequency version of `estimate_gamma_bc`.

Estimates the cost-function curvature γ from an annual balanced panel using IV
(with Δ(annual ISR) as instrument) followed by iterative model-based bias
correction that accounts for temporal aggregation from monthly to annual.

All non-γ structural parameters (c, fc, δ, β, ϵ, μν, σν2, Smax, Ns) are taken
as given from `params`.  The ω process parameters are re-estimated each
iteration from annual residuals and mapped to monthly frequency for the model
simulation.

# Arguments
- `params`    : Parameters object providing all fixed model parameters
- `df_annual` : Annual DataFrame with columns `firm_id`, `year_id`, `total_opex`,
                `total_sales`, `total_revenue`, `inv_to_sales`
- `n_years`   : years per firm in each bias-correction simulation (default 200)
- `n_firms`   : number of firms in each simulation (default 40)

# Returns
`(γ̂_BC, μω_monthly, σω2_monthly, ρω_monthly)` — bias-corrected γ and monthly
ω process parameters.
"""
function estimate_gamma_bc_annual(params::Parameters, df_annual::DataFrame;
                                   n_years::Int  = 200,
                                   n_firms::Int  = 40,
                                   max_iter::Int = 20,
                                   tol::Real     = 1e-2,
                                   seed::Int     = 212311)

    # --- Prepare annual regression DataFrame ---
    df_sorted = sort(df_annual, [:firm_id, :year_id])
    n_obs     = nrow(df_sorted)

    Δisr_vec          = fill(NaN, n_obs)
    firm_boundary_vec = falses(n_obs)
    firm_boundary_vec[1] = true
    for i in 2:n_obs
        if df_sorted.firm_id[i] != df_sorted.firm_id[i - 1]
            firm_boundary_vec[i] = true
        else
            Δisr_vec[i] = df_sorted.inv_to_sales[i] - df_sorted.inv_to_sales[i - 1]
        end
    end

    valid = (df_sorted.total_opex .> 0) .& (df_sorted.total_sales .> 0) .& .!isnan.(Δisr_vec)
    df_est = DataFrame(
        log_expense   = log.(df_sorted.total_opex[valid]),
        log_demand    = log.(df_sorted.total_sales[valid]),
        Δinv_sales    = Δisr_vec[valid],
        firm_boundary = firm_boundary_vec[valid]
    )

    # --- Step 1: initial IV estimate ---
    iv      = reg(df_est, @formula(log_expense ~ (log_demand ~ Δinv_sales)))
    γ̂_step1 = coef(iv)[end]

    log_ω_proxy = coef(iv)[1] .+ FixedEffectModels.residuals(iv, df_est)
    μω_ann, σω2_ann, ρω_ann = estimate_omega_ar1(log_ω_proxy, df_est.firm_boundary)
    μω_current, σω2_current, ρω_current = annual_ar1_to_monthly(μω_ann, σω2_ann, ρω_ann)

    println("\n=== Iterative Bias-Corrected Estimation (Annual Data) ===")
    println("Step 1 — Initial γ̂ (z-IV):  $(round(γ̂_step1, digits=6))")
    println("\n Iter │     γ̂        │    bias     │   γ̂_BC      │   μ̂_ω      │  σ̂η²(mo)   │  ρ̂_ω(mo)")
    println("──────┼──────────────┼─────────────┼─────────────┼─────────────┼─────────────┼───────────")

    γ̂_current = γ̂_step1
    γ̂_BC      = γ̂_current

    for iter in 1:max_iter
        # Reconstruct Parameters at current (γ, ω process), all other params from truth
        μ_ν_level  = exp(params.μν + 0.5 * params.σν2)
        σ_ν2_level = (exp(params.σν2) - 1.0) * μ_ν_level^2
        params_iter = Parameters(c=params.c, fc=params.fc,
                                  μω=μω_current, σω2=σω2_current, ρ_ω=ρω_current,
                                  γ=γ̂_current,
                                  δ=params.δ, β=params.β, ϵ=params.ϵ,
                                  μν=μ_ν_level, σν2=σ_ν2_level,
                                  Smax=params.Smax, Ns=params.Ns)
        _, _, _, _, ppi_iter, opi_iter, _ = solve_model(params_iter)
        Random.seed!(seed)
        n_months = n_years * 12
        _, _, dem_i, _, exp_i, ω_i, isr_i =
            simulate_firm(n_firms, n_months, ppi_iter, opi_iter, params_iter)

        # Model-implied bias at annual aggregation level
        bias_i = compute_annual_sim_bias(exp_i, dem_i, ω_i, isr_i, n_firms, n_months)

        # Re-estimate ω from annual residuals at current γ̂, map back to monthly
        log_ω_hat = df_est.log_expense .- γ̂_current .* df_est.log_demand
        μω_ann_new, σω2_ann_new, ρω_ann_new = estimate_omega_ar1(log_ω_hat, df_est.firm_boundary)
        μω_new, σω2_new, ρω_new = annual_ar1_to_monthly(μω_ann_new, σω2_ann_new, ρω_ann_new)

        γ̂_BC_new = γ̂_step1 - bias_i

        @printf("  %3d  │  %10.6f  │  %10.6f │  %10.6f │  %10.6f │  %10.6f │  %10.6f\n",
                iter, γ̂_current, bias_i, γ̂_BC_new, μω_new, σω2_new, ρω_new)

        converged   = abs(γ̂_BC_new - γ̂_BC) < tol
        γ̂_BC        = γ̂_BC_new
        γ̂_current   = γ̂_BC_new
        μω_current  = μω_new
        σω2_current = σω2_new
        ρω_current  = ρω_new

        if converged
            println("Converged at iteration $iter.")
            break
        end
    end

    println("\nFinal bias-corrected γ̂^BC: $(round(γ̂_BC, digits=6))")
    println("Final monthly ω params — μω: $(round(μω_current, digits=6))  " *
            "σω2: $(round(σω2_current, digits=6))  ρω: $(round(ρω_current, digits=6))")

    return γ̂_BC, μω_current, σω2_current, ρω_current
end
