"""
    estimate_omega_ar1(log_ω_proxy, firm_boundary)

Fit an AR(1) to a panel of log(ω) proxies and return the log-mean, innovation
variance, and persistence.  `log_ω_proxy` is a vector with observations stacked
across firms.  `firm_boundary` is a `Bool` vector of the same length whose `true`
entries mark the first observation of each firm (where no AR(1) lag exists).

Returns `(μη, ση2, ρω, se_μη, se_ση2, se_ρω)` where
- `μη`  = mean of the AR(1) innovation (intercept of log(ω) AR(1))  i.e.  a
- `ση2` = variance of the AR(1) innovation
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

    μη    = a                   # AR(1) intercept (mean of innovation η)
    ση2   = σ²_u                # innovation variance (= σ²_u)

    # SE for μη = a: directly the OLS SE for the intercept
    se_μη  = se_a

    # SE for ση2 = σ²_u: var of sample variance ≈ 2σ⁴/(T-2)
    se_ση2 = sqrt(2 * ση2^2 / max(T - 2, 1))

    return μη, ση2, ρω, se_μη, se_ση2, se_ρω
end


"""
    _compute_annual_auxiliary_arrays(tot_opex, tot_sales, tot_revenue, n_firms, n_years)

Fast array-based OLS + AR(1) computation used by `_simulate_all_moments`.
Avoids DataFrame construction and GLM overhead.

Data is assumed to be in firm-major order: rows 1:n_years belong to firm 1,
rows n_years+1:2*n_years to firm 2, etc. (the layout produced by
`_simulate_all_moments`).
"""
function _compute_annual_auxiliary_arrays(tot_opex::Vector{Float64},
                                          tot_sales::Vector{Float64},
                                          tot_revenue::Vector{Float64},
                                          n_firms::Int, n_years::Int)
    n = length(tot_opex)

    # Pass 1: means of log(sales) and log(opex) over valid obs
    n_v = 0; sum_lx = 0.0; sum_ly = 0.0
    @inbounds for i in 1:n
        tot_opex[i] > 0.0 && tot_sales[i] > 0.0 || continue
        n_v    += 1
        sum_lx += log(tot_sales[i])
        sum_ly += log(tot_opex[i])
    end
    x̄ = sum_lx / n_v
    ȳ = sum_ly / n_v

    # Pass 2: Sxx, Sxy → OLS slope
    Sxx = 0.0; Sxy = 0.0
    @inbounds for i in 1:n
        tot_opex[i] > 0.0 && tot_sales[i] > 0.0 || continue
        dx   = log(tot_sales[i]) - x̄
        Sxx += dx * dx
        Sxy += dx * (log(tot_opex[i]) - ȳ)
    end
    γ_OLS = Sxy / Sxx

    # Pass 3: build log_ω_proxy and firm_boundary for valid obs
    log_ω_proxy = Vector{Float64}(undef, n_v)
    firm_bnd    = falses(n_v)
    idx         = 0
    prev_firm   = -1
    @inbounds for i in 1:n
        tot_opex[i] > 0.0 && tot_sales[i] > 0.0 || continue
        idx += 1
        log_ω_proxy[idx] = log(tot_opex[i]) - γ_OLS * log(tot_sales[i])
        firm_i           = (i - 1) ÷ n_years + 1
        firm_bnd[idx]    = firm_i != prev_firm
        prev_firm        = firm_i
    end

    μ_η, σ_η2, ρ_ω, se_μη, se_ση2, se_ρω =
        estimate_omega_ar1(log_ω_proxy, firm_bnd)

    # Average operating expense / revenue ratio (annual)
    sum_opex_sales = 0.0
    n_os = 0
    @inbounds for i in 1:n
        tot_opex[i] > 0.0 && tot_revenue[i] > 0.0 || continue
        sum_opex_sales += tot_opex[i] / tot_revenue[i]
        n_os += 1
    end
    avg_opex_sales = sum_opex_sales / n_os

    return (γ_OLS=γ_OLS, ρ_ω=ρ_ω, σ_η2=σ_η2, μ_η=μ_η,
            avg_opex_sales=avg_opex_sales,
            se_ρω=se_ρω, se_ση2=se_ση2, se_μη=se_μη)
end


@inline function _radical_inverse(n::Int, base::Int)
    x = 0.0
    inv_base = 1.0 / base
    f = inv_base
    m = n
    while m > 0
        digit = m % base
        x += digit * f
        m = fld(m, base)
        f *= inv_base
    end
    return x
end


"""
    halton_param_vectors(bounds, n_points; seed=nothing)

Generate `n_points` parameter vectors using a randomized Halton sequence over
axis-aligned bounds.

- `bounds` must be a vector of `(low, high)` tuples, one for each dimension.
- Output is `Vector{Vector{Float64}}`, each inner vector matching the order in
  `bounds`.
- A random shift and random starting index are applied so draws are randomized
  but still space-filling.
"""
function halton_param_vectors(bounds::Vector{<:Tuple{<:Real,<:Real}},
                               n_points::Int;
                               seed::Union{Int,Nothing}=nothing)
    n_points > 0 || error("n_points must be positive")
    d = length(bounds)
    d > 0 || error("bounds must contain at least one dimension")

    # Enough primes for this project's parameter dimensions.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    d <= length(primes) || error("Increase prime list length for higher dimension Halton sampling")

    if !isnothing(seed)
        Random.seed!(seed)
    end

    lowers = Vector{Float64}(undef, d)
    spans  = Vector{Float64}(undef, d)
    for j in 1:d
        lo = Float64(bounds[j][1])
        hi = Float64(bounds[j][2])
        hi > lo || error("bounds[$j] must satisfy high > low")
        lowers[j] = lo
        spans[j]  = hi - lo
    end

    # Randomized Halton: random start index + Cranley-Patterson shift.
    start_index = rand(1:100_000)
    shifts = rand(d)

    points = Vector{Vector{Float64}}(undef, n_points)
    for i in 1:n_points
        x = Vector{Float64}(undef, d)
        n = start_index + i
        for j in 1:d
            u = _radical_inverse(n, primes[j])
            u_shift = mod(u + shifts[j], 1.0)
            x[j] = lowers[j] + spans[j] * u_shift
        end
        points[i] = x
    end

    return points
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
4. `μ̂_ω`   — unconditional log-mean of ω proxy

`df_annual` must have columns: `firm_id`, `year_id`, `total_opex`,
`total_sales`, `inv_to_sales`.

Returns a NamedTuple `(γ_OLS, ρ_ω, σ_η2, μ_η)`.
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
    γ_OLS       = coef(ols_result)[end]
    log_ω_proxy = coef(ols_result)[1] .+ residuals(ols_result)
    μ_η, σ_η2, ρ_ω, se_μη, se_ση2, se_ρω =
        estimate_omega_ar1(log_ω_proxy, df_ols.firm_boundary)

    # avg_opex_sales is defined as operating expenses divided by total revenue.
    valid_full = (df.total_opex .> 0) .& (df.total_revenue .> 0)
    avg_opex_sales = mean(df.total_opex[valid_full] ./ df.total_revenue[valid_full])

    return (γ_OLS=γ_OLS, ρ_ω=ρ_ω, σ_η2=σ_η2, μ_η=μ_η,
            avg_opex_sales=avg_opex_sales,
            se_ρω=se_ρω, se_ση2=se_ση2, se_μη=se_μη,
            ols_result=ols_result)
end


# ============================================================
# Full 7-parameter indirect inference estimator
# ============================================================

"""
    compute_monthly_moments(df_monthly)

Compute three moments from a monthly balanced panel:
1. `avg_isr`          — mean of BOM-inventory-to-revenue ratio
2. `var_log1p_isr`    — variance of `log(1 +` BOM-inventory-to-revenue ratio `)`
3. `avg_gross_margin` — mean of revenue / COGS  (= mean of p/c)

`df_monthly` must have columns `inv_to_sales`, `revenue`, `cogs`.
"""
function compute_monthly_moments(df_monthly::DataFrame)
    valid = (df_monthly.revenue .> 0)
    isr = df_monthly.inv_to_sales[valid]
    gm  = df_monthly.revenue[valid] ./ df_monthly.cogs[valid]
    log1p_isr = log1p.(isr)
    return (avg_isr=mean(isr), var_log1p_isr=var(log1p_isr), avg_gross_margin=mean(gm))
end


"""
    _simulate_all_moments(params, ppi, opi, n_firms, n_years, seed)

Simulate `n_firms` firms for `n_years * 12` months and return all seven moments
used by `estimate_params_ii_full`:

Monthly moments (computed from raw simulation output):
- `avg_isr`          — mean of BOM-inventory / revenue
- `var_log1p_isr`    — variance of `log(1 +` BOM-inventory / revenue `)`
- `avg_gross_margin` — mean of p/c

Annual auxiliary statistics (from `compute_annual_auxiliary`):
- `γ̂_OLS`, `ρ̂_ω`, `σ̂_η2`, `μ̂_ω`
"""
function _simulate_all_moments(params::Parameters, ppi, opi,
                                n_firms::Int, n_years::Int,
                                seed::Union{Int,Nothing})
    n_months = n_years * 12
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    inv_sim, dem_sim, exp_sim, rev_sim =
        simulate_firm(rng, n_firms, n_months, ppi, opi, params)

    # Single-pass monthly moments — no temporary arrays, Welford online variance
    n_mo = 0; sum_isr = 0.0; sum_gm = 0.0
    mean_l1p = 0.0; M2_l1p = 0.0
    any_above_grid = false
    @inbounds for k in eachindex(inv_sim)
        d = dem_sim[k]; r = rev_sim[k]; s = inv_sim[k]
        (d > 0.0 && r > 0.0) || continue
        isr      = s / r
        l1p      = log1p(isr)
        n_mo    += 1
        sum_isr += isr
        sum_gm  += r / (params.c * d)
        delta    = l1p - mean_l1p
        mean_l1p += delta / n_mo
        M2_l1p  += delta * (l1p - mean_l1p)   # Welford update
        s > params.Smax && (any_above_grid = true)
    end
    avg_isr_sim       = sum_isr / n_mo
    avg_gm_sim        = sum_gm  / n_mo
    var_log1p_isr_sim = M2_l1p  / (n_mo - 1)

    # Annual aggregation — @view avoids 12-element temporary slice allocations
    n_ann     = n_firms * n_years
    tot_opex    = Vector{Float64}(undef, n_ann)
    tot_sales   = Vector{Float64}(undef, n_ann)
    tot_revenue = Vector{Float64}(undef, n_ann)
    for firm in 1:n_firms
        m0 = (firm - 1) * n_months
        a0 = (firm - 1) * n_years
        for yr in 1:n_years
            m_first = m0 + (yr - 1) * 12 + 1
            m_last  = m0 + yr * 12
            a_idx   = a0 + yr
            tot_opex[a_idx]  = sum(@view exp_sim[m_first:m_last])
            tot_sales[a_idx] = sum(@view dem_sim[m_first:m_last])
            tot_revenue[a_idx] = sum(@view rev_sim[m_first:m_last])
        end
    end

    ψ̃_ann = _compute_annual_auxiliary_arrays(tot_opex, tot_sales, tot_revenue, n_firms, n_years)

    return (avg_isr=avg_isr_sim, var_log1p_isr=var_log1p_isr_sim, avg_gross_margin=avg_gm_sim,
            γ_OLS=ψ̃_ann.γ_OLS, ρ_ω=ψ̃_ann.ρ_ω, σ_η2=ψ̃_ann.σ_η2, avg_opex_sales=ψ̃_ann.avg_opex_sales,
            any_inventory_above_grid=any_above_grid)
end


"""
    compute_full_ii_target_moments(df_monthly, df_annual)

Compute the seven target moments used by the full indirect-inference objective.
Returns a NamedTuple with fields:
`avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, avg_opex_sales`.
"""
function compute_full_ii_target_moments(df_monthly::DataFrame,
                                         df_annual::DataFrame)
    mo_data  = compute_monthly_moments(df_monthly)
    ann_data = compute_annual_auxiliary(df_annual)
    return (
        avg_isr          = mo_data.avg_isr,
        var_log1p_isr    = mo_data.var_log1p_isr,
        avg_gross_margin = mo_data.avg_gross_margin,
        γ_OLS            = ann_data.γ_OLS,
        ρ_ω              = ann_data.ρ_ω,
        σ_η2             = ann_data.σ_η2,
        avg_opex_sales   = ann_data.avg_opex_sales
    )
end



@inline function _full_ii_target_vector(target_moments::NamedTuple)
    return [target_moments.avg_isr, target_moments.var_log1p_isr, target_moments.avg_gross_margin,
            target_moments.γ_OLS, target_moments.ρ_ω, target_moments.σ_η2, target_moments.avg_opex_sales]
end


@inline function _full_ii_parameter_vector(params::Parameters)
    return [params.γ, params.μη, params.ση2, params.ρ_ω, params.σν2, params.ϵ, params.δ]
end

@inline function _full_ii_params_from_vector(params_base::Parameters,
                                             θ::AbstractVector{<:Real})
    length(θ) == 7 || error("Parameter vector must have length 7")
    return Parameters(c=params_base.c, fc=params_base.fc,
                      μη=Float64(θ[2]), ση2=Float64(θ[3]), ρ_ω=Float64(θ[4]), γ=Float64(θ[1]),
                      δ=Float64(θ[7]), β=params_base.β, ϵ=Float64(θ[6]),
                      μν=params_base.μν, σν2=Float64(θ[5]),
                      Ns=params_base.Ns,
                      size=params_base.size)
end

function _full_ii_moment_vector_from_params(params::Parameters;
                                            n_firms::Int,
                                            n_years::Int,
                                            seed::Int,
                                            solve_maxiter::Int=1000)
    _, _, _, _, ppi, opi, _, converged = solve_model(params; maxiter=solve_maxiter)
    converged || error("solve_model did not converge")
    m̃_nt = _simulate_all_moments(params, ppi, opi, n_firms, n_years, seed)
    return _full_ii_target_vector(m̃_nt)
end


"""
    compute_full_ii_jacobian(params_base; n_firms, n_years, seed, step_sizes, solve_maxiter)

Numerically computes the 7×7 Jacobian of the simulated full-II moments with
respect to the 7 estimable parameters ordered as
`(γ, μη, ση2, ρω, σν2, ϵ, δ)`.

Rows follow the moment order
`(avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, μ_η)`.
"""
function compute_full_ii_jacobian(params_base::Parameters;
                                  n_firms::Int = 5000,
                                  n_years::Int = 20,
                                  seed::Int = 212311,
                                  solve_maxiter::Int = 1000)
    θ0 = _full_ii_parameter_vector(params_base)

    step = [max(abs(θ0[i]) * 1e-4, 1e-6) for i in 1:7]
    step[4] = max(step[4], 1e-5)
    step[7] = max(step[7], 1e-5)

    G = Matrix{Float64}(undef, 7, 7)

    for j in 1:7
        θ_plus = copy(θ0)
        θ_minus = copy(θ0)
        h = step[j]


            θ_plus[j] += h
            θ_minus[j] -= h
            m_plus = _full_ii_moment_vector_from_params(_full_ii_params_from_vector(params_base, θ_plus);
                                                        n_firms=n_firms, n_years=n_years,
                                                        seed=seed, solve_maxiter=solve_maxiter)
            m_minus = _full_ii_moment_vector_from_params(_full_ii_params_from_vector(params_base, θ_minus);
                                                         n_firms=n_firms, n_years=n_years,
                                                         seed=seed, solve_maxiter=solve_maxiter)
            G[:, j] = (m_plus - m_minus) ./ (2h)
    end

    return G
end


"""
    compute_full_ii_asymptotic_variance(params_base, W; n_firms, n_years, seed,
                                        solve_maxiter, sample_size)

Computes the efficient-GMM asymptotic variance by first numerically evaluating
the Jacobian of the simulated moments at `params_base`, then applying

    Avar(θ̂) = (G' W G)^(-1)

It returns both `Avar(θ̂)` and the finite-sample covariance approximation
`Avar(θ̂) / sample_size`, along with standard errors and the Jacobian used.
"""
function compute_full_ii_asymptotic_variance(params_base::Parameters,
                                             W::AbstractMatrix{<:Real};
                                             n_firms::Int = 5000,
                                             n_years::Int = 20,
                                             seed::Int = 212311,
                                             solve_maxiter::Int = 1000,
                                             sample_size::Int = 1)

    Gf = compute_full_ii_jacobian(params_base;
                                  n_firms=n_firms,
                                  n_years=n_years,
                                  seed=seed,
                                  solve_maxiter=solve_maxiter)
    Wf = Matrix{Float64}(W)
    avar = inv(Gf * Wf * Gf')
    vcov = avar / sample_size
    se = sqrt.(diag(vcov))

    return (
        G = Gf,
        avar = avar,
        vcov = vcov,
        se = se,
        parameter_names = (:γ, :μη, :ση2, :ρω, :σν2, :ϵ, :δ)
    )
end


"""
    select_best_grid_start(df_grid, target_moments; weighting_matrix)

Choose the parameter vector from a precomputed grid (for example,
`compute_moments_on_grid` output) that minimizes the same weighted objective
used in `estimate_params_ii_full`, evaluated against precomputed target
moments.

Returns a NamedTuple with fields
`row_index, obj_value, γ, μη, ση2, ρω, σν2, ϵ, δ`.
"""
function select_best_grid_start(df_grid::DataFrame,
                                 target_moments::NamedTuple,
                                 W::AbstractMatrix{<:Real})
                                 
    m̂ = _full_ii_target_vector(target_moments)

    best_idx = 0
    best_obj = Inf

    for i in 1:nrow(df_grid)
        if Bool(df_grid.failed[i])
            continue
        end

        m̃ = [df_grid.avg_isr[i], df_grid.var_log1p_isr[i], df_grid.avg_gross_margin[i],
              df_grid.γ_OLS[i],  df_grid.ρ_ω[i],           df_grid.σ_η2[i],
              df_grid.avg_opex_sales[i]]
        all(isfinite, m̃) || continue

        M = m̂ - m̃
        obj_value = dot(M, W * M)
        if obj_value < best_obj
            best_obj = obj_value
            best_idx = i
        end
    end

    best_idx > 0 || error("No valid candidate rows found in df_grid")

    return (
        row_index = best_idx,
        obj_value = best_obj,
        γ   = Float64(df_grid.γ[best_idx]),
        μη  = Float64(df_grid.μη[best_idx]),
        ση2 = Float64(df_grid.ση2[best_idx]),
        ρω  = Float64(df_grid.ρω[best_idx]),
        σν2 = Float64(df_grid.σν2[best_idx]),
        ϵ   = Float64(df_grid.ϵ[best_idx]),
        δ   = Float64(df_grid.δ[best_idx])
    )
end


"""
    estimate_params_ii_full(target_moments, init_guess, W; ...)

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
- Monthly: avg BOM-inventory/revenue ISR, variance of log(1 + ISR), avg gross margin (p/c)
- Annual: γ_OLS, ρ_ω, σ_η2, μ_η from the OLS auxiliary regression

**Objective** — quadratic form:

    obj(θ) = M(θ)' W M(θ)

where `M(θ) = m̂ − m̃(θ)` and `W` is a required 7×7 weighting matrix.

Minimised via Nelder-Mead over the unconstrained reparameterisation
`(γ, log μω, log σω2, arctanh ρω, log σν, ϵ, logit δ)`.

`init_guess` must be a length-7 vector ordered as
`[γ, μη, ση2, ρω, σν2, ϵ, δ]`.

Fixed parameters not estimated here are taken from `Parameters()` constructor
defaults.
`target_moments` must be a NamedTuple with fields:
`avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, μ_η`.

# Returns
Named tuple with fields `γ̂`, `μη`, `ση2`, `ρω`, `σν2`, `ϵ̂`, `δ̂`,
`obj_value`, `result`.
"""
function estimate_params_ii_full(target_moments::NamedTuple,init_guess::AbstractVector{<:Real},
                                  W::AbstractMatrix{<:Real};
                                  n_firms::Int   = 200,
                                  n_years::Int   = 50,
                                  γ_lb::Float64  = 0.05,  γ_ub::Float64  = 3.0,
                                  μη_lb::Float64 = -5.0,  μη_ub::Float64 =  5.0,
                                  σ2_lb::Float64 = 1e-6,  σ2_ub::Float64 = 5.0,
                                  ρ_lb::Float64  = -0.999, ρ_ub::Float64 = 0.999,
                                  σν2_lb::Float64 = 1e-6,  σν2_ub::Float64 = 5.0,
                                  ϵ_lb::Float64  = 1.1,   ϵ_ub::Float64  = 20.0,
                                  δ_lb::Float64  = 0.001, δ_ub::Float64  = 0.999,
                                  seed::Int      = 212311,
                                  max_iter::Int  = 1000,
                                  verbose::Bool  = true,
                                  g_abstol=1e-4)


    # --- Data moments ---
    m̂ = _full_ii_target_vector(target_moments)

    if verbose
        println("\n=== Full II Estimation — Data Moments ===")
        @printf("  avg_isr          = %10.6f\n", m̂[1])
        @printf("  var_log1p_isr    = %10.6f\n", m̂[2])
        @printf("  avg_gross_margin = %10.6f\n", m̂[3])
        @printf("  γ_OLS (annual)   = %10.6f\n", m̂[4])
        @printf("  ρ_ω   (annual)   = %10.6f\n", m̂[5])
        @printf("  σ_η2  (annual)   = %10.6f\n", m̂[6])
        @printf("  opex/revenue(ann)= %10.6f\n", m̂[7])
        println("\nStarting Nelder-Mead. Iteration output shows simulated moments and objective.")
        println("\n iter │ avg_isr │ var_log1p │ avg_gm  │  γ_OLS  │   ρ_ω   │   σ_η2  │ opx/rev │   obj")
        println("──────┼─────────┼───────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────")
    end

    iter_count = Ref(0)

    function unpack(x)
        γ_n   = clamp(x[1],                γ_lb,  γ_ub)
        μη_n  = clamp(x[2],                μη_lb, μη_ub)
        ση2_n = clamp(exp(x[3]),           σ2_lb, σ2_ub)
        ρω_n  = clamp(tanh(x[4]),          ρ_lb,  ρ_ub)
        σν2_n = clamp(exp(x[5]),           σν2_lb, σν2_ub)
        ϵ_n   = clamp(x[6],                ϵ_lb,  ϵ_ub)
        δ_n   = clamp(1/(1+exp(-x[7])),    δ_lb,  δ_ub)
        return γ_n, μη_n, ση2_n, ρω_n, σν2_n, ϵ_n, δ_n
    end

    function obj(x::Vector{Float64})
        iter_count[] += 1
        γ_n, μη_n, ση2_n, ρω_n, σν2_n, ϵ_n, δ_n = unpack(x)
        try
            params_iter = Parameters(μη=μη_n, ση2=ση2_n, ρ_ω=ρω_n, γ=γ_n,
                                      δ=δ_n,  ϵ=ϵ_n,
                                      σν2=σν2_n)
            _, _, _, _, ppi, opi, _, _ = solve_model(params_iter)
            m̃_nt = _simulate_all_moments(params_iter, ppi, opi, n_firms, n_years, seed)
            m̃ = [m̃_nt.avg_isr, m̃_nt.var_log1p_isr, m̃_nt.avg_gross_margin,
                m̃_nt.γ_OLS, m̃_nt.ρ_ω, m̃_nt.σ_η2, m̃_nt.avg_opex_sales]
            M = m̂ - m̃
            sse = dot(M, W * M)

            if verbose
                @printf("  %4d │ %7.4f │ %9.4f │ %7.4f │ %7.4f │ %7.4f │ %7.4f │ %7.4f │ %9.5f\n",
                    iter_count[], m̃[1], m̃[2], m̃[3], m̃[4], m̃[5], m̃[6], m̃[7], sse)
            end
            return sse
        catch
            verbose && @printf("  %4d — model failed, penalty returned\n", iter_count[])
            return 1e10
        end
    end

    # Initial point from params_base or user-supplied guess [γ, μη, ση2, ρω, σν2, ϵ, δ]

    length(init_guess) == 7 || error("init_guess must have length 7: [γ, μη, ση2, ρω, σν2, ϵ, δ]")
    γ_init, μη_init, ση2_init, ρω_init, σν2_init, ϵ_init, δ_init =
        Float64(init_guess[1]), Float64(init_guess[2]), Float64(init_guess[3]),
        Float64(init_guess[4]), Float64(init_guess[5]), Float64(init_guess[6]),
        Float64(init_guess[7])

    x0 = [clamp(γ_init,               γ_lb,  γ_ub),
          clamp(μη_init,              μη_lb, μη_ub),
          log(clamp(ση2_init,         σ2_lb,  σ2_ub)),
          atanh(clamp(ρω_init,        ρ_lb,  ρ_ub)),
          log(clamp(σν2_init,         σν2_lb, σν2_ub)),
          clamp(ϵ_init,               ϵ_lb,  ϵ_ub),
          log(clamp(δ_init,           δ_lb,  δ_ub) /
              (1.0 - clamp(δ_init,    δ_lb,  δ_ub)))]

    result = Optim.optimize(obj, x0, Optim.NelderMead(),
                             Optim.Options(iterations=max_iter, show_trace=false,
                                           x_abstol=1e-4, g_abstol=g_abstol))

    γ, μη_est, ση2_est, ρω_est, σν2_est, ϵ_est, δ_est = unpack(Optim.minimizer(result))

    if verbose
        println("\n=== Full II Estimation Complete ===")
        println("  Converged : $(Optim.converged(result))")
        @printf("  γ̂         = %10.6f\n", γ)
        @printf("  μ̂η        = %10.6f\n", μη_est)
        @printf("  σ̂²η       = %10.6f\n", ση2_est)
        @printf("  ρ̂ω        = %10.6f\n", ρω_est)
        @printf("  σ̂ν2       = %10.6f\n", σν2_est)
        @printf("  ϵ̂         = %10.6f\n", ϵ_est)
        @printf("  δ̂         = %10.6f\n", δ_est)
        println("  Objective : $(round(Optim.minimum(result), digits=8))")
    end

    return (γ=γ, μη=μη_est, ση2=ση2_est, ρω=ρω_est,
            σν2=σν2_est, ϵ=ϵ_est, δ=δ_est,
            obj_value=Optim.minimum(result), result=result)
end

"""
    compute_moments_on_grid(params_base, param_vectors;
                            n_firms, n_years, seed, output_path)

Evaluate all 7 estimation moments on an explicit list of parameter vectors.

Each element of `param_vectors` must be a length-7 vector ordered as
`[γ, μη, ση2, ρω, σν2, ϵ, δ]`.

For every parameter vector, the function solves the model, simulates
`n_firms × n_years × 12` months of panel data, and records the moments used in
`estimate_params_ii_full`. Results are written to a CSV file.

**Parallelisation:** the loop over grid points runs with `Threads.@threads`.
Start Julia with `julia --threads=N` (or set the environment variable
`JULIA_NUM_THREADS=N`) to exploit multiple cores.

- `param_vectors` : Vector of vectors, one per evaluation point, each with 7
                    entries in the order `[γ, μη, ση2, ρω, σν2, ϵ, δ]`.
- `n_firms`       : Firms simulated per parameter vector.
- `n_years`       : Years simulated per firm (months = `n_years × 12`).
- `seed`          : Random seed, held fixed across parameter vectors so moments are
                    comparable under the same simulation draws.
- `output_path`   : Path to the output CSV file.

# Returns
A `DataFrame` with parameter columns `γ, μη, ση2, ρω, σν, ϵ, δ` and moment
columns `avg_isr, var_log1p_isr, avg_gross_margin, γ̂_OLS, ρ̂_ω, σ̂_η2, μ̂_η, failed`.
`failed = true` indicates the model could not be solved at those parameters.
"""
function compute_moments_on_grid(param_vectors::AbstractVector{<:AbstractVector{<:Real}};
                                  n_firms::Int              = 5000,
                                  n_years::Int              = 20,
                                  seed::Int                 = 212311,
                                  max_value_iterations::Int = 500,
                                  output_path::String       = "grid_moments.csv")

    # --- Validate and convert user-supplied parameter vectors ---------------
    n_total = length(param_vectors)
    n_total > 0 || error("param_vectors must contain at least one parameter vector")

    combos = Vector{NTuple{7, Float64}}(undef, n_total)
    for i in eachindex(param_vectors)
        p = param_vectors[i]
        length(p) == 7 || error("param_vectors[$i] must have exactly 7 entries")
        combos[i] = (Float64(p[1]), Float64(p[2]), Float64(p[3]), Float64(p[4]),
                     Float64(p[5]), Float64(p[6]), Float64(p[7]))
    end


    @printf("Parameter sweep: %d total points  (%d threads available)\n",
            n_total, Threads.nthreads())

    # --- Pre-allocate result arrays -----------------------------------------
    out_γ    = Vector{Float64}(undef, n_total)
    out_μη   = Vector{Float64}(undef, n_total)
    out_ση2  = Vector{Float64}(undef, n_total)
    out_ρω   = Vector{Float64}(undef, n_total)
    out_σν2  = Vector{Float64}(undef, n_total)
    out_ϵ    = Vector{Float64}(undef, n_total)
    out_δ    = Vector{Float64}(undef, n_total)

    out_avg_isr  = fill(NaN, n_total)
    out_var_log1p_isr = fill(NaN, n_total)
    out_avg_gm   = fill(NaN, n_total)
    out_γ_ols    = fill(NaN, n_total)
    out_ρω_ar1   = fill(NaN, n_total)
    out_σ_η2            = fill(NaN, n_total)
    out_avg_opex_sales  = fill(NaN, n_total)
    out_failed          = fill(true, n_total)
    out_inventory_above_grid = fill(false, n_total)

    # --- Progress tracking --------------------------------------------------
    counter    = Threads.Atomic{Int}(0)
    print_lock = ReentrantLock()
    report_step = max(20, n_total ÷ 40)   # print at ~2.5 % increments
    t_start     = time()

    # --- Main parallel loop -------------------------------------------------
    Threads.@threads for idx in 1:n_total
        γ_i, μη_i, ση2_i, ρω_i, σν2_i, ϵ_i, δ_i = combos[idx]

        out_γ[idx]   = γ_i
        out_μη[idx]  = μη_i
        out_ση2[idx] = ση2_i
        out_ρω[idx]  = ρω_i
        out_σν2[idx] = σν2_i
        out_ϵ[idx]   = ϵ_i
        out_δ[idx]   = δ_i

        try
            params_i = Parameters(
                μη   = μη_i,
                ση2  = ση2_i,
                ρ_ω  = ρω_i,
                γ    = γ_i,
                δ    = δ_i,
                ϵ    = ϵ_i,
                σν2  = σν2_i)

            _, _, _, _, ppi, opi, _, converged_i = solve_model(params_i,maxiter=max_value_iterations)
            !converged_i && error("value function did not converge")
            row_seed = seed + idx - 1
            m̃ = _simulate_all_moments(params_i, ppi, opi, n_firms, n_years, row_seed)

            out_avg_isr[idx]  = m̃.avg_isr
            out_var_log1p_isr[idx]  = m̃.var_log1p_isr
            out_avg_gm[idx]   = m̃.avg_gross_margin
            out_γ_ols[idx]    = m̃.γ_OLS
            out_ρω_ar1[idx]          = m̃.ρ_ω
            out_σ_η2[idx]            = m̃.σ_η2
            out_avg_opex_sales[idx]  = m̃.avg_opex_sales
            out_inventory_above_grid[idx] = m̃.any_inventory_above_grid
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
                fmt_hms = s -> @sprintf("%02d:%02d:%02d", floor(Int,s/3600), floor(Int,rem(s,3600)/60), floor(Int,rem(s,60)))
                @printf("\r  [%s] %3.0f%%  %d/%d  (elapsed %s, ETA %s)   ",
                        bar, 100.0 * pct, done, n_total, fmt_hms(elapsed), fmt_hms(eta))
                flush(stdout)
                done == n_total && println()
            end
        end
    end

    # --- Assemble and save --------------------------------------------------
    df_out = DataFrame(
        γ                = out_γ,
        μη               = out_μη,
        ση2              = out_ση2,
        ρω               = out_ρω,
        σν2              = out_σν2,
        ϵ                = out_ϵ,
        δ                = out_δ,
        avg_isr          = out_avg_isr,
        var_log1p_isr    = out_var_log1p_isr,
        avg_gross_margin = out_avg_gm,
        γ_OLS            = out_γ_ols,
        ρ_ω              = out_ρω_ar1,
        σ_η2             = out_σ_η2,
        avg_opex_sales   = out_avg_opex_sales,
        any_inventory_above_grid = out_inventory_above_grid,
        failed           = out_failed)

    CSV.write(output_path, df_out)
    n_ok = sum(.!out_failed)
        frac_inventory_above_grid = n_ok > 0 ? mean(out_inventory_above_grid[.!out_failed]) : NaN
        @printf("\nGrid search complete.  %d / %d points succeeded.  Results → %s\n",
            n_ok, n_total, output_path)
        @printf("Fraction of successful runs with any simulated inventory above Smax: %.6f\n",
            frac_inventory_above_grid)

    return df_out
end