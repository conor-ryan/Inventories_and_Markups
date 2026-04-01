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
    μω_current, σω2_current, ρω_current, _, _, _ = estimate_omega_ar1(log_ω_proxy, df.firm_boundary)

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
        μ̂_ω_new, σ̂η2_new, ρ̂_ω_new, _, _, _ = estimate_omega_ar1(log_ω_hat, df.firm_boundary)

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
    T    = length(y)

    # OLS: y = a + ρ·x
    x̄, ȳ  = mean(x), mean(y)
    Sxx   = sum((x .- x̄).^2)
    ρω    = sum((x .- x̄) .* (y .- ȳ)) / Sxx
    a     = ȳ - ρω * x̄
    resid = y .- (a .+ ρω .* x)
    σ²_u  = sum(resid.^2) / (T - 2)          # OLS residual variance (df-corrected)

    # Standard errors of (a, ρω) from OLS sandwich
    se_ρω = sqrt(σ²_u / Sxx)
    se_a  = sqrt(σ²_u * (1/T + x̄^2 / Sxx))

    μω    = exp(a / (1 - ρω))   # unconditional mean level
    σω2   = σ²_u                # innovation variance (= σ²_u)

    # Delta-method SE for μω = exp(a/(1-ρω))
    # ∂μω/∂a  = μω / (1-ρω)
    # ∂μω/∂ρω = μω * a / (1-ρω)²
    dμ_da  = μω / (1 - ρω)
    dμ_dρ  = μω * a / (1 - ρω)^2
    # Approx (ignoring covariance of a and ρω — conservative)
    se_μω  = sqrt((dμ_da * se_a)^2 + (dμ_dρ * se_ρω)^2)

    # SE for σω2 = σ²_u: var of sample variance ≈ 2σ⁴/(T-2)
    se_σω2 = sqrt(2 * σω2^2 / max(T - 2, 1))

    return μω, σω2, ρω, se_μω, se_σω2, se_ρω
end


# ============================================================
# Indirect Inference estimation of (γ, μω, σω2, ρω) from
# annual panel data
# ============================================================

"""
    compute_annual_auxiliary(df_annual)

Compute four auxiliary statistics from an annual balanced panel:

1. `γ̂_OLS` — OLS estimate of γ: log(total_opex) ~ log(total_sales)
2. `ρ̂_ω`   — AR(1) persistence of annual log-ω proxy
3. `σ̂_η2`  — AR(1) innovation variance of annual log-ω proxy
4. `μ̂_ω`   — unconditional level mean of ω proxy

`df_annual` must have columns: `firm_id`, `year_id`, `total_opex`,
`total_sales`, `inv_to_sales`.

Returns a NamedTuple `(γ̂_OLS, ρ̂_ω, σ̂_η2, μ̂_ω)`.
"""
function compute_annual_auxiliary(df_annual::DataFrame)
    df = sort(df_annual, [:firm_id, :year_id])
    n  = nrow(df)

    firm_bnd_vec = falses(n)
    firm_bnd_vec[1] = true
    for i in 2:n
        if df.firm_id[i] != df.firm_id[i - 1]
            firm_bnd_vec[i] = true
        end
    end

    valid = (df.total_opex .> 0) .& (df.total_sales .> 0)
    df_ols = DataFrame(
        log_opex      = log.(df.total_opex[valid]),
        log_sales     = log.(df.total_sales[valid]),
        firm_boundary = firm_bnd_vec[valid]
    )

    ols_result  = lm(@formula(log_opex ~ log_sales), df_ols)
    γ̂_OLS       = coef(ols_result)[end]
    log_ω_proxy = coef(ols_result)[1] .+ residuals(ols_result)
    μ̂_ω, σ̂_η2, ρ̂_ω, se_μω, se_σω2, se_ρω =
        estimate_omega_ar1(log_ω_proxy, df_ols.firm_boundary)

    return (γ̂_OLS=γ̂_OLS, ρ̂_ω=ρ̂_ω, σ̂_η2=σ̂_η2, μ̂_ω=μ̂_ω,
            se_ρω=se_ρω, se_σω2=se_σω2, se_μω=se_μω,
            ols_result=ols_result)
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
    ψ̂_vec = [ψ̂.γ̂_OLS, ψ̂.ρ̂_ω, ψ̂.σ̂_η2, ψ̂.μ̂_ω]   # se_* not used in objective
    # Normalisation: weight inversely proportional to |ψ̂_k|²
    w_vec = [1.0 / max(abs(v), 1e-8)^2 for v in ψ̂_vec]

    if verbose
        println("\n=== Indirect Inference: Annual Data Auxiliary Statistics ===")
        @printf("  γ̂_OLS = %10.6f\n",  ψ̂.γ̂_OLS)
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
            ψ̃_vec = [ψ̃.γ̂_OLS, ψ̃.ρ̂_ω, ψ̃.σ̂_η2, ψ̃.μ̂_ω]
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
                             Optim.Options(iterations=max_iter, show_trace=false,
                                           x_abstol=1e-4, g_abstol=1e-4))

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


# ============================================================
# Full 7-parameter indirect inference estimator
# ============================================================

"""
    compute_monthly_moments(df_monthly)

Compute three moments from a monthly balanced panel:
1. `avg_isr`          — mean of BOM-inventory-to-revenue ratio
2. `var_isr`          — variance of BOM-inventory-to-revenue ratio
3. `avg_gross_margin` — mean of revenue / COGS  (= mean of p/c)

`df_monthly` must have columns `inv_to_sales`, `revenue`, `cogs`.
"""
function compute_monthly_moments(df_monthly::DataFrame)
    valid = (df_monthly.inv_to_sales .> 0) .& isfinite.(df_monthly.inv_to_sales) .&
            (df_monthly.revenue .> 0) .& (df_monthly.cogs .> 0)
    isr = df_monthly.inv_to_sales[valid]
    gm  = df_monthly.revenue[valid] ./ df_monthly.cogs[valid]
    return (avg_isr=mean(isr), var_isr=var(isr), avg_gross_margin=mean(gm))
end


"""
    _simulate_all_moments(params, ppi, opi, n_firms, n_years, seed)

Simulate `n_firms` firms for `n_years * 12` months and return all seven moments
used by `estimate_params_ii_full`:

Monthly moments (computed from raw simulation output):
- `avg_isr`          — mean of BOM-inventory / revenue
- `var_isr`          — variance of BOM-inventory / revenue
- `avg_gross_margin` — mean of p/c

Annual auxiliary statistics (from `compute_annual_auxiliary`):
- `γ̂_OLS`, `ρ̂_ω`, `σ̂_η2`, `μ̂_ω`
"""
function _simulate_all_moments(params::Parameters, ppi, opi,
                                n_firms::Int, n_years::Int,
                                seed::Union{Int,Nothing})
    n_months = n_years * 12
    if !isnothing(seed)
        Random.seed!(seed)
    end

    inv_sim, _, dem_sim, _, exp_sim, _, isr_sim =
        simulate_firm(n_firms, n_months, ppi, opi, params)

    # Monthly moments
    # isr_sim[t] = c·s_t/(p_t·D_t)  →  BOM/revenue ISR = s_t/(p_t·D_t) = isr_sim[t]/c
    # gross margin = p/c = s_t / (isr_sim[t]·D_t)   (requires D_t > 0)
    valid_mo = (isr_sim .> 0) .& (dem_sim .> 0) .& isfinite.(isr_sim)
    isr_mo   = isr_sim[valid_mo] ./ params.c
    gm_mo    = inv_sim[valid_mo] ./ (isr_sim[valid_mo] .* dem_sim[valid_mo])
    avg_isr_sim = mean(isr_mo)
    var_isr_sim = var(isr_mo)
    avg_gm_sim  = mean(gm_mo)

    # Annual aggregation for auxiliary regression
    n_ann    = n_firms * n_years
    firm_ids = Vector{Int}(undef, n_ann)
    year_ids = Vector{Int}(undef, n_ann)
    tot_opex  = Vector{Float64}(undef, n_ann)
    tot_sales = Vector{Float64}(undef, n_ann)

    for firm in 1:n_firms
        m0 = (firm - 1) * n_months
        a0 = (firm - 1) * n_years
        for yr in 1:n_years
            m_first = m0 + (yr - 1) * 12 + 1
            m_last  = m0 + yr * 12
            a_idx   = a0 + yr
            firm_ids[a_idx]  = firm
            year_ids[a_idx]  = yr
            tot_opex[a_idx]  = sum(exp_sim[m_first:m_last])
            tot_sales[a_idx] = sum(dem_sim[m_first:m_last])
        end
    end

    df_ann = DataFrame(firm_id=firm_ids, year_id=year_ids,
                       total_opex=tot_opex, total_sales=tot_sales)
    ψ̃_ann = compute_annual_auxiliary(df_ann)

    return (avg_isr=avg_isr_sim, var_isr=var_isr_sim, avg_gross_margin=avg_gm_sim,
            γ̂_OLS=ψ̃_ann.γ̂_OLS, ρ̂_ω=ψ̃_ann.ρ̂_ω, σ̂_η2=ψ̃_ann.σ̂_η2, μ̂_ω=ψ̃_ann.μ̂_ω)
end


"""
    estimate_params_ii_full(params_base, df_monthly, df_annual; ...)

Indirect inference estimator for all seven estimable structural parameters:

| Parameter | Description                        | Identifies          |
|-----------|------------------------------------|---------------------|
| γ         | cost-function curvature            | annual OLS slope    |
| μω        | level mean of cost shock ω         | annual AR(1) mean   |
| σω2       | innovation variance of log(ω)      | annual AR(1) σ²     |
| ρω        | AR(1) persistence of log(ω)        | annual AR(1) ρ      |
| σν        | log-space std of demand shock ν    | monthly ISR level   |
| ϵ         | demand elasticity                  | gross margin        |
| δ         | inventory depreciation rate        | monthly ISR variance|

**Data moments** (7 total):
- Monthly: avg BOM-inventory/revenue ISR, variance of ISR, avg gross margin (p/c)
- Annual: γ̂_OLS, ρ̂_ω, σ̂²_η, μ̂_ω  from the OLS auxiliary regression

**Objective** — normalised SSE:

    obj(θ) = Σ_k  [(m̂_k − m̃_k(θ)) / |m̂_k|]²

Minimised via Nelder-Mead over the unconstrained reparameterisation
`(γ, log μω, log σω2, arctanh ρω, log σν, ϵ, logit δ)`.

The level mean of ν (μν) is held fixed at its value in `params_base`.

# Returns
Named tuple with fields `γ̂`, `μω`, `σω2`, `ρω`, `σν`, `ϵ̂`, `δ̂`,
`obj_value`, `result`.
"""
function estimate_params_ii_full(params_base::Parameters,
                                  df_monthly::DataFrame,
                                  df_annual::DataFrame;
                                  n_firms::Int   = 200,
                                  n_years::Int   = 50,
                                  γ_lb::Float64  = 0.05,  γ_ub::Float64  = 3.0,
                                  μω_lb::Float64 = 0.01,  μω_ub::Float64 = 100.0,
                                  σ2_lb::Float64 = 1e-6,  σ2_ub::Float64 = 5.0,
                                  ρ_lb::Float64  = -0.999, ρ_ub::Float64 = 0.999,
                                  σν_lb::Float64 = 1e-4,  σν_ub::Float64 = 5.0,
                                  ϵ_lb::Float64  = 1.1,   ϵ_ub::Float64  = 20.0,
                                  δ_lb::Float64  = 0.001, δ_ub::Float64  = 0.999,
                                  seed::Int      = 212311,
                                  max_iter::Int  = 1000,
                                  verbose::Bool  = true)

    # Fixed level mean of ν  (recovered from params_base log-space fields)
    μν_level = exp(params_base.μν + 0.5 * params_base.σν2)

    # --- Data moments ---
    mo_data  = compute_monthly_moments(df_monthly)
    ann_data = compute_annual_auxiliary(df_annual)
    m̂ = [mo_data.avg_isr, mo_data.var_isr, mo_data.avg_gross_margin,
          ann_data.γ̂_OLS, ann_data.ρ̂_ω, ann_data.σ̂_η2, ann_data.μ̂_ω]
    w = [1.0 / max(abs(v), 1e-8)^2 for v in m̂]

    if verbose
        println("\n=== Full II Estimation — Data Moments ===")
        @printf("  avg_isr          = %10.6f\n", m̂[1])
        @printf("  var_isr          = %10.6f\n", m̂[2])
        @printf("  avg_gross_margin = %10.6f\n", m̂[3])
        @printf("  γ̂_OLS (annual)   = %10.6f\n", m̂[4])
        @printf("  ρ̂_ω  (annual)    = %10.6f\n", m̂[5])
        @printf("  σ̂²_η (annual)    = %10.6f\n", m̂[6])
        @printf("  μ̂_ω  (annual)    = %10.6f\n", m̂[7])
        println("\nStarting Nelder-Mead over (γ, log μω, log σ²ω, arctanh ρω, log σν, ϵ, logit δ)...")
        println("\n iter │    γ    │    μω   │   σ²ω   │    ρω   │    σν   │    ϵ    │    δ    │   obj")
        println("──────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────")
    end

    iter_count = Ref(0)

    function unpack(x)
        γ_n   = clamp(x[1],                γ_lb,  γ_ub)
        μω_n  = clamp(exp(x[2]),           μω_lb, μω_ub)
        σω2_n = clamp(exp(x[3]),           σ2_lb, σ2_ub)
        ρω_n  = clamp(tanh(x[4]),          ρ_lb,  ρ_ub)
        σν_n  = clamp(exp(x[5]),           σν_lb, σν_ub)
        ϵ_n   = clamp(x[6],                ϵ_lb,  ϵ_ub)
        δ_n   = clamp(1/(1+exp(-x[7])),    δ_lb,  δ_ub)
        return γ_n, μω_n, σω2_n, ρω_n, σν_n, ϵ_n, δ_n
    end

    function obj(x::Vector{Float64})
        iter_count[] += 1
        γ_n, μω_n, σω2_n, ρω_n, σν_n, ϵ_n, δ_n = unpack(x)
        try
            # Reconstruct level variance of ν from fixed level mean and new log-space σν
            σν2_level_n = μν_level^2 * (exp(σν_n^2) - 1.0)
            params_iter = Parameters(c=params_base.c, fc=params_base.fc,
                                      μω=μω_n, σω2=σω2_n, ρ_ω=ρω_n, γ=γ_n,
                                      δ=δ_n, β=params_base.β, ϵ=ϵ_n,
                                      μν=μν_level, σν2=σν2_level_n,
                                      Smax=params_base.Smax, Ns=params_base.Ns)
            _, _, _, _, ppi, opi, _ = solve_model(params_iter)
            m̃_nt = _simulate_all_moments(params_iter, ppi, opi, n_firms, n_years, seed)
            m̃ = [m̃_nt.avg_isr, m̃_nt.var_isr, m̃_nt.avg_gross_margin,
                  m̃_nt.γ̂_OLS,  m̃_nt.ρ̂_ω,   m̃_nt.σ̂_η2, m̃_nt.μ̂_ω]
            sse = sum(w[k] * (m̂[k] - m̃[k])^2 for k in 1:7)

            if verbose
                @printf("  %4d │ %7.4f │ %7.4f │ %7.5f │ %7.4f │ %7.4f │ %7.3f │ %7.4f │ %9.5f\n",
                        iter_count[], γ_n, μω_n, σω2_n, ρω_n, σν_n, ϵ_n, δ_n, sse)
            end
            return sse
        catch
            verbose && @printf("  %4d — model failed, penalty returned\n", iter_count[])
            return 1e10
        end
    end

    # Initial point from params_base
    σν_init = params_base.σν   # log-space std already stored in params
    x0 = [params_base.γ,
          log(clamp(exp(params_base.μω),  μω_lb, μω_ub)),
          log(clamp(params_base.σω2,       σ2_lb, σ2_ub)),
          atanh(clamp(params_base.ρ_ω,    ρ_lb,  ρ_ub)),
          log(clamp(σν_init,              σν_lb, σν_ub)),
          clamp(params_base.ϵ,            ϵ_lb,  ϵ_ub),
          log(clamp(params_base.δ, δ_lb, δ_ub) /
              (1.0 - clamp(params_base.δ, δ_lb, δ_ub)))]

    result = Optim.optimize(obj, x0, Optim.NelderMead(),
                             Optim.Options(iterations=max_iter, show_trace=false,
                                           x_abstol=1e-4, g_abstol=1e-4))

    γ̂, μω_est, σω2_est, ρω_est, σν_est, ϵ_est, δ_est = unpack(Optim.minimizer(result))

    if verbose
        println("\n=== Full II Estimation Complete ===")
        println("  Converged : $(Optim.converged(result))")
        @printf("  γ̂         = %10.6f\n", γ̂)
        @printf("  μ̂ω        = %10.6f\n", μω_est)
        @printf("  σ̂²ω       = %10.6f\n", σω2_est)
        @printf("  ρ̂ω        = %10.6f\n", ρω_est)
        @printf("  σ̂ν        = %10.6f\n", σν_est)
        @printf("  ϵ̂         = %10.6f\n", ϵ_est)
        @printf("  δ̂         = %10.6f\n", δ_est)
        println("  Objective : $(round(Optim.minimum(result), digits=8))")
    end

    return (γ̂=γ̂, μω=μω_est, σω2=σω2_est, ρω=ρω_est,
            σν=σν_est, ϵ̂=ϵ_est, δ̂=δ_est,
            obj_value=Optim.minimum(result), result=result)
end


"""
    compute_moments_on_grid(params_base; n_grid, γ_range, μω_range, σω2_range,
                            ρω_range, σν_range, ϵ_range, δ_range,
                            n_firms, n_years, seed, output_path)

Evaluate all 7 estimation moments on a Cartesian grid of parameter values.

For every combination of (γ, μω, σω2, ρω, σν, ϵ, δ) grid points the function
solves the model, simulates `n_firms × n_years × 12` months of panel data, and
records the moments used in `estimate_params_ii_full`.  Results are written to a
CSV file.

**Parallelisation:** the loop over grid points runs with `Threads.@threads`.
Start Julia with `julia --threads=N` (or set the environment variable
`JULIA_NUM_THREADS=N`) to exploit multiple cores.

# Arguments
- `params_base`   : `Parameters` object; `c`, `fc`, `β`, `μν`, `Smax`, `Ns`
                    are held fixed at their values here.
- `n_grid`        : Grid points per parameter.  Either a single `Int` (same for
                    all 7 parameters) or a length-7 `Vector{Int}` in the order
                    `[γ, μω, σω2, ρω, σν, ϵ, δ]`.
- `γ_range`       : `(lb, ub)` for the γ grid.
- `μω_range`      : `(lb, ub)` for the μω grid (**level** mean of ω; the
                    constructor converts this to log-space internally).
- `σω2_range`     : `(lb, ub)` for the σω2 grid (level innovation variance).
- `ρω_range`      : `(lb, ub)` for the ρω grid.
- `σν_range`      : `(lb, ub)` for σν (**log-space** std-dev of ν, stored as
                    `params.σν`).
- `ϵ_range`       : `(lb, ub)` for the demand-elasticity grid.
- `δ_range`       : `(lb, ub)` for the depreciation-rate grid.
- `n_firms`       : Firms simulated per grid point.
- `n_years`       : Years simulated per firm (months = `n_years × 12`).
- `seed`          : Random seed, held fixed across grid points so moments are
                    comparable under the same simulation draws.
- `output_path`   : Path to the output CSV file.

# Returns
A `DataFrame` with parameter columns `γ, μω, σω2, ρω, σν, ϵ, δ` and moment
columns `avg_isr, var_isr, avg_gross_margin, γ̂_OLS, ρ̂_ω, σ̂_η2, μ̂_ω, failed`.
`failed = true` indicates the model could not be solved at those parameters.
"""
function compute_moments_on_grid(params_base::Parameters;
                                  n_grid                    = 5,
                                  γ_range::Tuple            = (0.5, 1.5),
                                  μω_range::Tuple           = (0.05, 2.0),
                                  σω2_range::Tuple          = (0.01, 0.3),
                                  ρω_range::Tuple           = (0.0, 0.9),
                                  σν_range::Tuple           = (0.1, 1.5),
                                  ϵ_range::Tuple            = (2.0, 15.0),
                                  δ_range::Tuple            = (0.01, 0.3),
                                  n_firms::Int              = 200,
                                  n_years::Int              = 20,
                                  seed::Int                 = 212311,
                                  output_path::String       = "grid_moments.csv")

    # --- Expand n_grid to a length-7 vector ---------------------------------
    ng = (n_grid isa Int) ? fill(n_grid, 7) : collect(Int, n_grid)
    length(ng) == 7 || error("n_grid must be an Int or a length-7 Vector{Int}")

    # --- Build individual parameter grids -----------------------------------
    γ_grid   = collect(LinRange(Float64(γ_range[1]),   Float64(γ_range[2]),   ng[1]))
    μω_grid  = collect(LinRange(Float64(μω_range[1]),  Float64(μω_range[2]),  ng[2]))
    σω2_grid = collect(LinRange(Float64(σω2_range[1]), Float64(σω2_range[2]), ng[3]))
    ρω_grid  = collect(LinRange(Float64(ρω_range[1]),  Float64(ρω_range[2]),  ng[4]))
    σν_grid  = collect(LinRange(Float64(σν_range[1]),  Float64(σν_range[2]),  ng[5]))
    ϵ_grid   = collect(LinRange(Float64(ϵ_range[1]),   Float64(ϵ_range[2]),   ng[6]))
    δ_grid   = collect(LinRange(Float64(δ_range[1]),   Float64(δ_range[2]),   ng[7]))

    # Fixed level mean of ν (held constant across grid points)
    μν_level = exp(params_base.μν + 0.5 * params_base.σν2)

    # --- Enumerate all Cartesian combinations --------------------------------
    combos  = collect(Iterators.product(γ_grid, μω_grid, σω2_grid,
                                         ρω_grid, σν_grid, ϵ_grid, δ_grid))
    n_total = length(combos)

    @printf("Grid search: %d total points  (%d threads available)\n",
            n_total, Threads.nthreads())
    @printf("Grid sizes : γ=%d  μω=%d  σω2=%d  ρω=%d  σν=%d  ϵ=%d  δ=%d\n", ng...)

    # --- Pre-allocate result arrays -----------------------------------------
    out_γ    = Vector{Float64}(undef, n_total)
    out_μω   = Vector{Float64}(undef, n_total)
    out_σω2  = Vector{Float64}(undef, n_total)
    out_ρω   = Vector{Float64}(undef, n_total)
    out_σν   = Vector{Float64}(undef, n_total)
    out_ϵ    = Vector{Float64}(undef, n_total)
    out_δ    = Vector{Float64}(undef, n_total)

    out_avg_isr  = fill(NaN, n_total)
    out_var_isr  = fill(NaN, n_total)
    out_avg_gm   = fill(NaN, n_total)
    out_γ_ols    = fill(NaN, n_total)
    out_ρω_ar1   = fill(NaN, n_total)
    out_σ_η2     = fill(NaN, n_total)
    out_μω_ar1   = fill(NaN, n_total)
    out_failed   = fill(true, n_total)

    # --- Progress tracking --------------------------------------------------
    counter    = Threads.Atomic{Int}(0)
    print_lock = ReentrantLock()
    report_step = max(1, n_total ÷ 40)   # print at ~2.5 % increments
    t_start     = time()

    # --- Main parallel loop -------------------------------------------------
    Threads.@threads for idx in 1:n_total
        γ_i, μω_i, σω2_i, ρω_i, σν_i, ϵ_i, δ_i = combos[idx]

        out_γ[idx]   = γ_i
        out_μω[idx]  = μω_i
        out_σω2[idx] = σω2_i
        out_ρω[idx]  = ρω_i
        out_σν[idx]  = σν_i
        out_ϵ[idx]   = ϵ_i
        out_δ[idx]   = δ_i

        try
            σν2_level_i = μν_level^2 * (exp(σν_i^2) - 1.0)
            params_i = Parameters(
                c    = params_base.c,
                fc   = params_base.fc,
                μω   = μω_i,          # constructor expects level mean
                σω2  = σω2_i,
                ρ_ω  = ρω_i,
                γ    = γ_i,
                δ    = δ_i,
                β    = params_base.β,
                ϵ    = ϵ_i,
                μν   = μν_level,
                σν2  = σν2_level_i,
                Smax = params_base.Smax,
                Ns   = params_base.Ns)

            _, _, _, _, ppi, opi, _ = solve_model(params_i)
            m̃ = _simulate_all_moments(params_i, ppi, opi, n_firms, n_years, seed)

            out_avg_isr[idx]  = m̃.avg_isr
            out_var_isr[idx]  = m̃.var_isr
            out_avg_gm[idx]   = m̃.avg_gross_margin
            out_γ_ols[idx]    = m̃.γ̂_OLS
            out_ρω_ar1[idx]   = m̃.ρ̂_ω
            out_σ_η2[idx]     = m̃.σ̂_η2
            out_μω_ar1[idx]   = m̃.μ̂_ω
            out_failed[idx]   = false
        catch
            # Leave NaN outputs and failed = true
        end

        # --- Progress bar ---------------------------------------------------
        done = Threads.atomic_add!(counter, 1) + 1
        if done % report_step == 0 || done == n_total
            lock(print_lock) do
                elapsed = time() - t_start
                pct     = done / n_total
                eta     = pct < 1.0 ? elapsed / pct * (1.0 - pct) : 0.0
                bar_len = 40
                filled  = round(Int, bar_len * pct)
                bar     = "=" ^ max(0, filled - 1) * ">" * " " ^ (bar_len - filled)
                @printf("\r  [%s] %3.0f%%  %d/%d  (%.0fs elapsed, %.0fs ETA)   ",
                        bar, 100.0 * pct, done, n_total, elapsed, eta)
                flush(stdout)
                done == n_total && println()
            end
        end
    end

    # --- Assemble and save --------------------------------------------------
    df_out = DataFrame(
        γ                = out_γ,
        μω               = out_μω,
        σω2              = out_σω2,
        ρω               = out_ρω,
        σν               = out_σν,
        ϵ                = out_ϵ,
        δ                = out_δ,
        avg_isr          = out_avg_isr,
        var_isr          = out_var_isr,
        avg_gross_margin = out_avg_gm,
        γ_OLS            = out_γ_ols,
        ρ_ω              = out_ρω_ar1,
        σ_η2             = out_σ_η2,
        μ_ω              = out_μω_ar1,
        failed           = out_failed)

    CSV.write(output_path, df_out)
    n_ok = sum(.!out_failed)
    @printf("\nGrid search complete.  %d / %d points succeeded.  Results → %s\n",
            n_ok, n_total, output_path)

    return df_out
end