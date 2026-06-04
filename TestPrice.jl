# TestPrice.jl
# Optimal pricing in a simplified model: no inventories, no stockouts.
#
# Model:  D = ν · p^{−ϵ}  (deterministic demand shock ν, never inventory-capped)
#         cost = ω · D^γ + c · D
#         π(p) = p · D − ω · D^γ − c · D
#
# The firm's first-order condition gives the markup formula:
#         p = (ϵ / (ϵ−1)) · [ω · γ · ν^{γ−1} · p^{ϵ(1−γ)} + c]
# which is the νbar → ∞ (no stockout) limit of price_residual in ModelFunctions.jl.
#
# Usage:
include("ModelFunctions.jl")
#   include("TestPrice.jl")
#   params  = Parameters(γ=1.0, ϵ=3.0, c=0.5)
#   ν_grid  = collect(range(0.5, 5.0, length=100))
#   γ_grid  = [0.7, 0.85, 1.0, 1.15, 1.3]
#   ω       = 1.0
#   P_star, plt_p = test_price(ν_grid, γ_grid, ω, params)
#   Π_star, plt_π = test_profit(ν_grid, γ_grid, ω, params, P_star)

using Optim, Plots

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

"""
    price_residual_simple(p, ν, ω, ϵ, γ, c) -> Float64

FOC residual for the no-inventory, no-stockout pricing problem.

Profit π = p·D − ω·D^γ − c·D with D = ν·p^{−ϵ}.  The first-order condition
yields the markup formula

    p = (ϵ/(ϵ−1)) · [ω·γ·ν^{γ−1}·p^{ϵ(1−γ)} + c]

Returns p minus the right-hand side.  This is the νbar → ∞ limit of
`price_residual` in ModelFunctions.jl: the truncated-lognormal ratio
E[ν^γ | ν < νbar] / E[ν | ν < νbar] collapses to ν^{γ−1} and the
stockout correction term (tail / Fbar) vanishes.
"""
function price_residual_simple(p::Float64, ν::Float64, ω::Float64,
                                ϵ::Float64, γ::Float64, c::Float64)::Float64
    opp_mc = ω * γ * ν^(γ - 1.0) * p^(ϵ * (1.0 - γ))
    rhs    = (ϵ / (ϵ - 1.0)) * (opp_mc + c)
    return p - rhs
end

"""
    solve_price_simple(ν, ω, ϵ, γ, c; p_lo, p_hi) -> (p_opt, converged)

Find the optimal price for given deterministic demand shock ν and cost shock ω.
Minimises the squared FOC residual via Brent's method, mirroring
`solve_price_policy` in ModelFunctions.jl.
"""
function solve_price_simple(ν::Float64, ω::Float64, ϵ::Float64, γ::Float64, c::Float64;
                             p_lo::Float64=1e-4, p_hi::Float64=1e4)::Tuple{Float64,Bool}
    # When ϵ(1−γ) > 1 the residual f = p − h(p) rises from negative to a peak
    # then falls to −∞.  The profit-maximising price is the SMALLEST zero of f,
    # which lies on the upward slope.  The peak of f occurs at the analytical
    # critical point p₁ (where f′ = 0), giving a tight upper bracket for that
    # smallest zero.  On [p_lo, p₁] f is monotone so Brent on f² finds it reliably.
    if ϵ * (1.0 - γ) > 1.0
        exponent = ϵ * (1.0 - γ) - 1.0   # > 0
        p_upper  = ((ϵ - 1.0) / (ϵ^2 * (1.0 - γ) * ω * γ * ν^(γ - 1.0)))^(1.0 / exponent)
        p_upper  = min(p_upper, p_hi)
    else
        p_upper  = p_hi
    end

    obj(p) = price_residual_simple(p, ν, ω, ϵ, γ, c)^2
    result    = Optim.optimize(obj, p_lo, p_upper, Brent(), rel_tol=1e-14, abs_tol=1e-12)
    p_opt     = result.minimizer
    converged = Optim.converged(result) && sqrt(result.minimum) < 1e-6
    return p_opt, converged
end

"""
    profit_simple(p, ν, ω, ϵ, γ, c) -> Float64

Profit at price p:  π = p·D − ω·D^γ − c·D,  D = ν·p^{−ϵ}.
"""
function profit_simple(p::Float64, ν::Float64, ω::Float64,
                        ϵ::Float64, γ::Float64, c::Float64)::Float64
    D = ν * p^(-ϵ)
    return p * D - ω * D^γ - c * D
end

# ---------------------------------------------------------------------------
# Grid-sweep and plotting functions
# ---------------------------------------------------------------------------

"""
    test_price(ν_grid, γ_grid, ω, params; p_lo, p_hi) -> (P_star, plt)

Solve for the optimal price across a grid of demand shocks `ν_grid` and
returns-to-scale values `γ_grid` at a fixed cost shock `ω`.  Uses `params.ϵ`
and `params.c`; all other fields of `params` are ignored (no dynamic
programming).

Returns:
- `P_star` : `Nν × Nγ` matrix of optimal prices
- `plt`    : Plots figure — optimal price vs ν, one line per γ level
"""
function test_price(ν_grid::AbstractVector{Float64}, γ_grid::AbstractVector{Float64},
                    ω::Float64, params::Parameters;
                    p_lo::Float64=1e-4, p_hi::Float64=1e4)
    Nν = length(ν_grid)
    Nγ = length(γ_grid)
    ϵ  = params.ϵ
    c  = params.c

    P_star        = zeros(Nν, Nγ)
    all_converged = true

    for (j, γ) in enumerate(γ_grid)
        for (i, ν) in enumerate(ν_grid)
            p_opt, conv    = solve_price_simple(ν, ω, ϵ, γ, c; p_lo=p_lo, p_hi=p_hi)
            P_star[i, j]   = p_opt
            all_converged  = all_converged && conv
        end
    end

    if !all_converged
        @warn "test_price: some price solutions did not converge"
    end

    plt = plot(ν_grid, P_star[:, 1],
               label  = "γ = $(round(γ_grid[1], digits=2))",
               xlabel = "ν  (demand shock)",
               ylabel = "Optimal price  p*",
               title  = "Optimal price — no-inventory model  (ω = $ω, c = $(round(c, digits=3)))",
               legend = :topright,
               linewidth = 2)
    for j in 2:Nγ
        plot!(plt, ν_grid, P_star[:, j],
              label     = "γ = $(round(γ_grid[j], digits=2))",
              linewidth = 2)
    end
    display(plt)

    return P_star, plt
end

"""
    test_profit(ν_grid, γ_grid, ω, params, P_star) -> (Π_star, plt)

Compute and plot the profit at the optimal prices stored in `P_star`.
`P_star[i, j]` should be the optimal price for `(ν_grid[i], γ_grid[j])`,
as returned by `test_price`.

Returns:
- `Π_star` : `Nν × Nγ` matrix of optimal profits
- `plt`    : Plots figure — profit vs ν, one line per γ level
"""
function test_profit(ν_grid::AbstractVector{Float64}, γ_grid::AbstractVector{Float64},
                     ω::Float64, params::Parameters, P_star::Matrix{Float64})
    Nν = length(ν_grid)
    Nγ = length(γ_grid)
    ϵ  = params.ϵ
    c  = params.c

    Π_star = zeros(Nν, Nγ)

    for (j, γ) in enumerate(γ_grid)
        for (i, ν) in enumerate(ν_grid)
            Π_star[i, j] = profit_simple(P_star[i, j], ν, ω, ϵ, γ, c)
        end
    end

    plt = plot(ν_grid, Π_star[:, 1],
               label  = "γ = $(round(γ_grid[1], digits=2))",
               xlabel = "ν  (demand shock)",
               ylabel = "Profit at p*",
               title  = "Profit at optimal price — no-inventory model  (ω = $ω, c = $(round(c, digits=3)))",
               legend = :topright,
               linewidth = 2)
    for j in 2:Nγ
        plot!(plt, ν_grid, Π_star[:, j],
              label     = "γ = $(round(γ_grid[j], digits=2))",
              linewidth = 2)
    end
    display(plt)

    return Π_star, plt
end

# ---------------------------------------------------------------------------
# Run: parameters from SolveModel.jl
# ---------------------------------------------------------------------------

include("ModelFunctions.jl")

params = Parameters(c=1.0, fc=0.0, μη=log(0.027), ση2=0.05, ρ_ω=0.1, γ=0.9,
                    δ=0.05, β=0.995, ϵ=16.0, μν=1, σν2=0.05, Ns=400,
                    scale=1.0, size=200.0)

# ν grid: 5th–95th percentile of the model's demand shock distribution
ν_lo   = quantile(params.dist, 0.05)
ν_hi   = quantile(params.dist, 0.95)
ν_grid = collect(range(ν_lo, ν_hi, length=100))

# γ grid: sweep from γ_lo to γ_hi
γ_lo   = 0.1
γ_hi   = 1.5
γ_n    = 15
γ_grid = collect(range(γ_lo, γ_hi, length=γ_n))

# ω: median of the model's Tauchen cost-shock grid
ω = 0.001

P_star, plt_p = test_price(ν_grid, γ_grid, ω, params)
Π_star, plt_π = test_profit(ν_grid, γ_grid, ω, params, P_star)
