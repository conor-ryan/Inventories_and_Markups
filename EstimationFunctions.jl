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
