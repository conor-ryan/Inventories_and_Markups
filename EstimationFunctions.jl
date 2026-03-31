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


# ============================================================
# Indirect Inference estimation of (γ, μω, σω2, ρω) from
# annual panel data
# ============================================================

"""
    compute_annual_auxiliary(df_annual)

Compute four auxiliary statistics from an annual balanced panel:

1. `γ̂_IV`  — IV estimate of γ: log(total_opex) ~ log(total_sales),
              instrument = Δ(inv_to_sales)
2. `ρ̂_ω`   — AR(1) persistence of annual log-ω proxy
3. `σ̂_η2`  — AR(1) innovation variance of annual log-ω proxy
4. `μ̂_ω`   — unconditional level mean of ω proxy

`df_annual` must have columns: `firm_id`, `year_id`, `total_opex`,
`total_sales`, `inv_to_sales`.

Returns a NamedTuple `(γ̂_IV, ρ̂_ω, σ̂_η2, μ̂_ω)`.
"""
function compute_annual_auxiliary(df_annual::DataFrame)
    df = sort(df_annual, [:firm_id, :year_id])
    n  = nrow(df)

    Δisr_vec     = fill(NaN, n)
    firm_bnd_vec = falses(n)
    firm_bnd_vec[1] = true
    for i in 2:n
        if df.firm_id[i] != df.firm_id[i - 1]
            firm_bnd_vec[i] = true
        else
            Δisr_vec[i] = df.inv_to_sales[i] - df.inv_to_sales[i - 1]
        end
    end

    valid = (df.total_opex .> 0) .& (df.total_sales .> 0) .& .!isnan.(Δisr_vec)
    df_iv = DataFrame(
        log_opex      = log.(df.total_opex[valid]),
        log_sales     = log.(df.total_sales[valid]),
        Δisr          = Δisr_vec[valid],
        firm_boundary = firm_bnd_vec[valid]
    )

    iv_result  = reg(df_iv, @formula(log_opex ~ (log_sales ~ Δisr)))
    γ̂_IV       = coef(iv_result)[end]
    log_ω_proxy = coef(iv_result)[1] .+ FixedEffectModels.residuals(iv_result, df_iv)
    μ̂_ω, σ̂_η2, ρ̂_ω = estimate_omega_ar1(log_ω_proxy, df_iv.firm_boundary)

    return (γ̂_IV=γ̂_IV, ρ̂_ω=ρ̂_ω, σ̂_η2=σ̂_η2, μ̂_ω=μ̂_ω)
end


"""
    _simulate_and_get_annual(params, ppi, opi, n_firms, n_years, seed)

Simulate `n_firms` firms for `n_years * 12` months using the supplied policy
interpolants `ppi` and `opi`, then aggregate to an annual panel DataFrame with
columns `firm_id`, `year_id`, `total_opex`, `total_sales`, `inv_to_sales`.

`inv_to_sales` is defined as BOY inventory divided by average monthly revenue
over the year, matching the definition in `simulate_panel_data`.
"""
function _simulate_and_get_annual(params::Parameters, ppi, opi,
                                   n_firms::Int, n_years::Int,
                                   seed::Union{Int,Nothing})
    n_months = n_years * 12
    if !isnothing(seed)
        Random.seed!(seed)
    end

    inv_sim, _, dem_sim, _, exp_sim, _, isr_sim =
        simulate_firm(n_firms, n_months, ppi, opi, params)

    n_ann    = n_firms * n_years
    firm_ids = Vector{Int}(undef, n_ann)
    year_ids = Vector{Int}(undef, n_ann)
    tot_opex = Vector{Float64}(undef, n_ann)
    tot_sales = Vector{Float64}(undef, n_ann)
    isr_ann  = Vector{Float64}(undef, n_ann)

    for firm in 1:n_firms
        m0 = (firm - 1) * n_months
        a0 = (firm - 1) * n_years
        for yr in 1:n_years
            m_first = m0 + (yr - 1) * 12 + 1
            m_last  = m0 + yr * 12
            a_idx   = a0 + yr

            # Monthly revenue: isr_sim[t] = c·s_t/(p_t·D_t), so p_t·D_t = c·s_t/isr_sim[t]
            ann_rev = sum(isr_sim[t] > 0 ? params.c * inv_sim[t] / isr_sim[t] : 0.0
                          for t in m_first:m_last)
            avg_monthly_rev  = ann_rev / 12

            firm_ids[a_idx]  = firm
            year_ids[a_idx]  = yr
            tot_opex[a_idx]  = sum(exp_sim[m_first:m_last])
            tot_sales[a_idx] = sum(dem_sim[m_first:m_last])
            isr_ann[a_idx]   = avg_monthly_rev > 0.0 ? inv_sim[m_first] / avg_monthly_rev : NaN
        end
    end

    return DataFrame(
        firm_id      = firm_ids,
        year_id      = year_ids,
        total_opex   = tot_opex,
        total_sales  = tot_sales,
        inv_to_sales = isr_ann
    )
end


"""
    estimate_params_ii_annual(params_base, df_annual; ...)

Indirect inference estimator for `(γ, μω_monthly, σω2_monthly, ρω_monthly)`
from an annual balanced panel.

**Auxiliary model** — applied identically to the data and to each simulation:
1. IV regression: `log(total_opex) ~ log(total_sales)`, instrument = `Δ(inv_to_sales)`
   → `γ̂_IV`
2. AR(1) fitted within-firm to the annual log-ω proxy from IV residuals
   → `(μ̂_ω, σ̂_η2, ρ̂_ω)` at annual frequency

**Objective** — normalised SSE between data and simulated auxiliary statistics:

    obj(θ) = Σ_k  [(ψ̂_k − ψ̃_k(θ)) / |ψ̂_k|]²

Minimised via Nelder-Mead over the unconstrained reparameterisation
`(γ, log μω, log σω2, arctanh ρω)`.

All non-estimated structural parameters are taken from `params_base`.

# Returns
`NamedTuple` with fields `γ̂`, `μω_monthly`, `σω2_monthly`, `ρω_monthly`,
`obj_value`, `result`.
"""
function estimate_params_ii_annual(params_base::Parameters, df_annual::DataFrame;
                                    n_firms::Int   = 200,
                                    n_years::Int   = 50,
                                    γ_lb::Float64  = 0.05,
                                    γ_ub::Float64  = 3.0,
                                    μω_lb::Float64 = 0.01,
                                    μω_ub::Float64 = 100.0,
                                    σ2_lb::Float64 = 1e-6,
                                    σ2_ub::Float64 = 5.0,
                                    ρ_lb::Float64  = -0.999,
                                    ρ_ub::Float64  =  0.999,
                                    seed::Int      = 212311,
                                    max_iter::Int  = 500,
                                    verbose::Bool  = true)

    # --- Step 0: auxiliary statistics from the data ---
    ψ̂ = compute_annual_auxiliary(df_annual)
    ψ̂_vec = [ψ̂.γ̂_IV, ψ̂.ρ̂_ω, ψ̂.σ̂_η2, ψ̂.μ̂_ω]
    # Normalisation: weight inversely proportional to |ψ̂_k|²
    w_vec = [1.0 / max(abs(v), 1e-8)^2 for v in ψ̂_vec]

    if verbose
        println("\n=== Indirect Inference: Annual Data Auxiliary Statistics ===")
        @printf("  γ̂_IV  = %10.6f\n",  ψ̂.γ̂_IV)
        @printf("  ρ̂_ω   = %10.6f  (annual)\n", ψ̂.ρ̂_ω)
        @printf("  σ̂²_η  = %10.6f  (annual)\n", ψ̂.σ̂_η2)
        @printf("  μ̂_ω   = %10.6f  (level)\n",  ψ̂.μ̂_ω)
        println("\nStarting Nelder-Mead over (γ, log μω, log σ²ω, arctanh ρω)...")
        println("\n iter │      γ      │    μω_mo    │   σ²ω_mo    │   ρω_mo    │  obj")
        println("──────┼─────────────┼─────────────┼─────────────┼────────────┼─────────────")
    end

    iter_count = Ref(0)

    # Map unconstrained x → bounded structural parameters
    function unpack(x)
        γ_n   = clamp(x[1],        γ_lb,  γ_ub)
        μω_n  = clamp(exp(x[2]),   μω_lb, μω_ub)
        σω2_n = clamp(exp(x[3]),   σ2_lb, σ2_ub)
        ρω_n  = clamp(tanh(x[4]),  ρ_lb,  ρ_ub)
        return γ_n, μω_n, σω2_n, ρω_n
    end

    function obj(x::Vector{Float64})
        iter_count[] += 1
        γ_n, μω_n, σω2_n, ρω_n = unpack(x)
        try
            μ_ν_level  = exp(params_base.μν + 0.5 * params_base.σν2)
            σ_ν2_level = (exp(params_base.σν2) - 1.0) * μ_ν_level^2
            params_iter = Parameters(c=params_base.c, fc=params_base.fc,
                                      μω=μω_n, σω2=σω2_n, ρ_ω=ρω_n, γ=γ_n,
                                      δ=params_base.δ, β=params_base.β, ϵ=params_base.ϵ,
                                      μν=μ_ν_level, σν2=σ_ν2_level,
                                      Smax=params_base.Smax, Ns=params_base.Ns)
            _, _, _, _, ppi, opi, _ = solve_model(params_iter)
            df_sim = _simulate_and_get_annual(params_iter, ppi, opi, n_firms, n_years, seed)
            ψ̃ = compute_annual_auxiliary(df_sim)
            ψ̃_vec = [ψ̃.γ̂_IV, ψ̃.ρ̂_ω, ψ̃.σ̂_η2, ψ̃.μ̂_ω]
            sse = sum(w_vec[k] * (ψ̂_vec[k] - ψ̃_vec[k])^2 for k in 1:4)

            if verbose
                @printf("  %4d │  %9.5f  │  %9.5f  │  %9.6f  │  %8.5f  │  %11.6f\n",
                        iter_count[], γ_n, μω_n, σω2_n, ρω_n, sse)
            end
            return sse
        catch
            verbose && @printf("  %4d — model failed, penalty returned\n", iter_count[])
            return 1e10
        end
    end

    # Initial point from params_base (μω stored as log-mean → exponentiate for level)
    γ_init   = params_base.γ
    μω_init  = exp(params_base.μω)
    σω2_init = params_base.σω2
    ρω_init  = params_base.ρ_ω
    x0 = [γ_init,
          log(clamp(μω_init,  μω_lb, μω_ub)),
          log(clamp(σω2_init, σ2_lb, σ2_ub)),
          atanh(clamp(ρω_init, ρ_lb, ρ_ub))]

    result = Optim.optimize(obj, x0, Optim.NelderMead(),
                             Optim.Options(iterations=max_iter, show_trace=false))

    γ̂, μω_est, σω2_est, ρω_est = unpack(Optim.minimizer(result))

    if verbose
        println("\n=== Indirect Inference Estimation Complete ===")
        println("  Converged : $(Optim.converged(result))")
        @printf("  γ̂         = %10.6f\n", γ̂)
        @printf("  μ̂ω (mo)   = %10.6f\n", μω_est)
        @printf("  σ̂²ω (mo)  = %10.6f\n", σω2_est)
        @printf("  ρ̂ω (mo)   = %10.6f\n", ρω_est)
        println("  Objective : $(round(Optim.minimum(result), digits=8))")
    end

    return (γ̂=γ̂, μω_monthly=μω_est, σω2_monthly=σω2_est, ρω_monthly=ρω_est,
            obj_value=Optim.minimum(result), result=result)
end


# --- DEAD CODE (kept for reference) ---
# The functions below used the iterative bias-correction approach for annual data.
# Replaced by indirect inference; see estimate_params_ii_annual above.

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
