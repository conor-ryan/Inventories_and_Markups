using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations, Random

# ---------------------------------------------------
# Parameters struct (matches SolveModel.jl)
# ---------------------------------------------------
struct Parameters
    c::Float64
    fc::Float64
    μη::Float64               # AR(1) innovation mean (intercept of log(ω) AR(1) process)
    ση2::Float64              # AR(1) innovation variance σ_η^2
    ρ_ω::Float64              # AR(1) persistence of log(ω)
    Q_ω::Int64                # number of omega grid points
    ω_grid::Vector{Float64}   # Qω omega values in levels
    P_ω::Matrix{Float64}      # Qω × Qω Tauchen transition matrix (row = current state)
    π_ω::Vector{Float64}      # ergodic stationary distribution over omega grid
    γ::Float64
    δ::Float64
    β::Float64
    ϵ::Float64
    μν::Float64               # level mean of demand shock ν
    σν2::Float64              # level variance of demand shock ν
    dist::LogNormal
    Q::Int64
    quad_nodes::Vector{Float64}
    quad_weights::Vector{Float64}
    quad_nodes_lognormal::Vector{Float64}
    gl_nodes::Vector{Float64}     # Gauss-Legendre nodes on [-1, 1]
    gl_weights::Vector{Float64}   # Gauss-Legendre weights
    Smax::Float64
    Ns::Int64
    Sgrid::Vector{Float64}
    size::Float64

    function Parameters(; c=1.0, fc=0.0, μη=0.0, ση2=0.0, ρ_ω=0.9, γ=1.0, δ=0.2, β=0.95, ϵ=2.0, μν=100, σν2=2832, Q=19, Q_ω=7, scale=1.0, size=1.0,  Ns=800)
                
        x, w = gausshermite(Q)
        gl_x, gl_w = gausslegendre(Q)
        


        scale_parameter = scale^(ϵ)
        c = c * scale
        μν = μν * scale_parameter * size
        σν2 = σν2 * scale_parameter^2 * size^2
        μη = μη + (1-ρ_ω)*log(scale) + (1-ρ_ω)*log(size^(1-γ))

        # Compute log-space parameters from mean and variance of ν
        σ2 = log(1 + σν2 / μν^2)
        σ  = sqrt(σ2)
        μ  = log(μν) - 0.5 * σ2

        # Transform quadrature nodes to lognormal draws (ν-space)
        x_lognormal = [exp(μ + sqrt(2) * σ * x[i]) for i in 1:Q]

        # μη is the AR(1) intercept (mean of innovation η); unconditional log-mean of ω = μη / (1 − ρ_ω)
        μ_log_ω = abs(1 - ρ_ω) > 1e-10 ? μη / (1 - ρ_ω) : μη
        σ_η     = sqrt(max(ση2, 0.0))

        # Tauchen discretization of the AR(1) in log(ω)
        if ση2 <= 0.0 || Q_ω <= 1
            ω_grid_vec = [exp(μ_log_ω)]   # single grid point at the unconditional level mean
            P_ω_mat    = ones(1, 1)
            π_ω_vec    = [1.0]
        else
            σ_logω   = σ_η / sqrt(1.0 - ρ_ω^2)   # unconditional std of log(ω)
            m_tau    = 3.0
            log_ω_lo = μ_log_ω - m_tau * σ_logω
            log_ω_hi = μ_log_ω + m_tau * σ_logω
            log_ω_grid = collect(range(log_ω_lo, log_ω_hi, length=Q_ω))
            h          = log_ω_grid[2] - log_ω_grid[1]
            ω_grid_vec = exp.(log_ω_grid)

            P_ω_mat = zeros(Q_ω, Q_ω)
            for i in 1:Q_ω
                cond_mean = μ_log_ω + ρ_ω * (log_ω_grid[i] - μ_log_ω)
                for j in 1:Q_ω
                    if j == 1
                        P_ω_mat[i, j] = cdf(Normal(), (log_ω_grid[j] + h / 2 - cond_mean) / σ_η)
                    elseif j == Q_ω
                        P_ω_mat[i, j] = 1.0 - cdf(Normal(), (log_ω_grid[j] - h / 2 - cond_mean) / σ_η)
                    else
                        P_ω_mat[i, j] = cdf(Normal(), (log_ω_grid[j] + h / 2 - cond_mean) / σ_η) -
                                         cdf(Normal(), (log_ω_grid[j] - h / 2 - cond_mean) / σ_η)
                    end
                end
                P_ω_mat[i, :] ./= sum(P_ω_mat[i, :])   # normalize row
            end

            # Ergodic stationary distribution via power iteration
            π_ω_vec = ones(Q_ω) / Q_ω
            for _ in 1:2000
                π_ω_vec = P_ω_mat' * π_ω_vec
            end
            π_ω_vec ./= sum(π_ω_vec)
        end



        μν = μν / (scale_parameter * size)
        σν2 = σν2 / (scale_parameter^2 * size^2)
        demand_dist = LogNormal(μ, σ)

        # Create inventory state grid
        Smax=quantile(demand_dist,0.9)*(ϵ - 1)/ϵ 
        Sgrid_vec = collect(range(1e-4, Smax, length=Ns))

        new(c, fc, μη, ση2, ρ_ω, length(ω_grid_vec), ω_grid_vec, P_ω_mat, π_ω_vec,
            γ, δ, β, ϵ, μν, σν2,demand_dist , Q, x, w, x_lognormal, gl_x, gl_w, Smax, Ns, Sgrid_vec,size)
    end
end

"""
    solve_model(params; verbose=false)
    
Solve the complete model: price policy, value function, and order policy.
Returns (p_policy, order_policy, V, price_policy_interp, order_policy_interp, Vinterp)
"""
function solve_model(params; full=false, verbose=false,fast_interp=true,maxiter=1000)
    Sgrid  = params.Sgrid
    ω_grid = params.ω_grid
    Nω     = params.Q_ω

    if verbose
        println("Solving value function...")
    end
    V, order_policy, p_policy, V_by_omega, converged = solve_value_function(params, full=full,fast_interp=fast_interp,maxiter=maxiter)

    price_policy_interp_nodes, order_policy_interp_nodes =
        build_fast_policy_interpolants(Sgrid, p_policy, order_policy)

    price_policy_interp = OmegaPolicyInterp(price_policy_interp_nodes, ω_grid)
    order_policy_interp = OmegaPolicyInterp(order_policy_interp_nodes, ω_grid)

    Vinterp = LinearInterpolation(Sgrid, V, extrapolation_bc=Line())

    return p_policy, order_policy, V, V_by_omega, price_policy_interp, order_policy_interp, Vinterp, converged
end

"""
    UniformInterp

Lightweight callable for linear interpolation (+ linear extrapolation) on a
**uniform** grid.  Much faster than `Interpolations.LinearInterpolation` in
tight inner loops because the grid index is computed directly (one multiply +
one add) rather than via binary search, and the call overhead is minimal.

# Fields
- `V`    : value vector (length Ns)
- `s_lo` : grid lower bound (`Sgrid[1]`)
- `inv_h`: `(Ns-1) / (Sgrid[end] - Sgrid[1])`  (precomputed reciprocal step)
"""
struct UniformInterp
    V    :: Vector{Float64}
    s_lo :: Float64
    inv_h :: Float64
end

@inline function (f::UniformInterp)(x::Float64)::Float64
    t = (x - f.s_lo) * f.inv_h      # fractional 0-based index
    n = length(f.V)
    if t <= 0.0
        @inbounds return f.V[1] + t * (f.V[2] - f.V[1])
    elseif t >= n - 1
        excess = t - (n - 1)
        @inbounds return f.V[n] + excess * (f.V[n] - f.V[n-1])
    else
        i = floor(Int, t) + 1        # 1-based lower index
        α = t - (i - 1)              # fractional part in [0, 1)
        @inbounds return f.V[i] + α * (f.V[i+1] - f.V[i])
    end
end

struct OmegaPolicyInterp
    nodes :: Vector{UniformInterp}
    ω_grid :: Vector{Float64}
end

@inline function (f::OmegaPolicyInterp)(x::Float64, ω_idx::Int)::Float64
    @inbounds return f.nodes[ω_idx](x)
end

@inline function (f::OmegaPolicyInterp)(x::Float64, ω::Float64)::Float64
    j = argmin(abs.(f.ω_grid .- ω))
    @inbounds return f.nodes[j](x)
end

function build_fast_policy_interpolants(Sgrid::AbstractVector{<:Real},
                                        p_policy::AbstractMatrix{<:Real},
                                        order_policy::AbstractMatrix{<:Real})
    inv_h = (length(Sgrid) - 1) / (Sgrid[end] - Sgrid[1])
    s_lo = Float64(Sgrid[1])
    price_nodes = [UniformInterp(Vector{Float64}(p_policy[:, j]), s_lo, inv_h) for j in axes(p_policy, 2)]
    order_nodes = [UniformInterp(Vector{Float64}(order_policy[:, j]), s_lo, inv_h) for j in axes(order_policy, 2)]
    return price_nodes, order_nodes
end


"""
    update_parameters(params::Parameters, x::Vector{Float64})
    
Create a new Parameters object with updated depreciation rate (δ), 
demand variance (σν2), and demand elasticity (ϵ).

Takes:
- params: Existing Parameters object
- x: Vector of length 3 containing [δ, σν2, ϵ]

Returns a new Parameters object with updated values.
"""
function update_parameters(params::Parameters, x::Vector{Float64})
    if length(x) != 3
        error("Input vector must have length 3: [δ, σν2, ϵ]")
    end
    
    δ_new = x[1]
    σν2_new = exp(x[2])
    ϵ_new = x[3]
    
    # params.μν stores the level mean of ν; pass it directly to the constructor
    μ_params = params.μν

    # Create new Parameters object
    return Parameters(
        c   = params.c,
        fc  = params.fc,
        μη  = params.μη,
        ση2 = params.ση2,
        ρ_ω = params.ρ_ω,
        γ   = params.γ,
        δ   = δ_new,
        β   = params.β,
        ϵ   = ϵ_new,
        μν  = μ_params,
        σν2 = σν2_new,
        Q   = params.Q,
        Q_ω = params.Q_ω,
        Ns   = params.Ns,
        size = params.size
    )
end


"""
    solve_price_policy(params)
    
Solve the price policy using the residual equation.
Returns the price policy vector.
"""
function solve_price_policy(params::Parameters, c_tilde::Float64, ω::Float64)
    Sgrid = params.Sgrid
    Ns = params.Ns
    p_policy = zeros(Ns)
    
    for (i, s) in enumerate(Sgrid)

        obj(p) = price_residual(p, s, c_tilde, ω, params)^2
        result = Optim.optimize(obj, 1e-3, 50.0, Brent(), rel_tol=1e-16, abs_tol=1e-12)
        p_policy[i] = result.minimizer
    end
    
    return p_policy
end


"""
    truncated_lognormal_ratio_Eνγ_Eν(νbar, params)

Compute the ratio E[ν^γ | ν < νbar] / E[ν | ν < νbar] for ν ~ LogNormal(μ, σ)
using Gauss-Legendre quadrature adapted to the truncated support.

The key idea is the probability-integral transform: let u = F(ν), so ν = F^{-1}(u).
The truncation ν < νbar maps to u ∈ [0, Fbar].  Gauss-Legendre nodes on [-1, 1]
are rescaled to [0, Fbar], giving integration points that respond continuously to
changes in νbar (unlike fixed Gauss-Hermite nodes with an indicator).

The GL change of variables u = (Fbar/2)(t + 1) introduces a Jacobian of Fbar/2.
This cancels in the ratio but must be retained for the individual expectations:

    E[ν^γ | ν < νbar] = (1/2) Σ_q w_q ν_q^γ
    E[ν   | ν < νbar] = (1/2) Σ_q w_q ν_q
    ratio              = (Σ_q w_q ν_q^γ) / (Σ_q w_q ν_q)

where ν_q = exp(μ + σ * Φ^{-1}((Fbar/2) * (gl_node_q + 1))).

Returns a NamedTuple (ratio, Eνγ, Eν).
"""
function truncated_lognormal_ratio_Eνγ_Eν(νbar::Float64, params::Parameters)
    μ_log = params.dist.μ
    σ_log = params.dist.σ
    Fbar  = cdf(params.dist, νbar)

    if Fbar < 1e-10
        return (ratio=0.0, Eνγ=0.0, Eν=0.0)
    end

    half_Fbar = 0.5 * Fbar
    num = 0.0
    den = 0.0
    for q in 1:params.Q
        # Map GL node from [-1,1] to (0, Fbar)
        u_q = half_Fbar * (params.gl_nodes[q] + 1.0)
        ν_q = exp(μ_log + σ_log * quantile(Normal(), u_q))
        w_q = params.gl_weights[q]
        num += w_q * ν_q^params.γ
        den += w_q * ν_q
    end

    ratio = den > 0.0 ? num / den : 0.0
    return (ratio=ratio, Eνγ=0.5 * num, Eν=0.5 * den)
end


"""
    price_residual(p, s, params)

Compute pricing residual matching SolveModel.jl
"""
function price_residual(p, s, c_tilde, ω, params)
    νbar = s * p^params.ϵ

    Fbar = cdf(params.dist, νbar)
    tail = 1.0 - Fbar

    # avoid numerical issues
    if Fbar < 1e-10
        return 1e6
    end

    Eν = truncated_lognormal_mean(νbar, params)

    ratio_Eνγ = truncated_lognormal_ratio_Eνγ_Eν(νbar, params).ratio

    opp_mc = ω * params.γ * ratio_Eνγ * p^(params.ϵ*(1-params.γ))
    rhs = (params.ϵ / (params.ϵ - 1)) *(opp_mc +  c_tilde) +
          (1 / (params.ϵ - 1)) *
          s * p^(params.ϵ + 1) *
          (1 / Eν) *
          (tail / Fbar)

    err = p - rhs

    return err
end


"""
    solve_value_function(p_policy, params; tol=1e-6, maxiter=1000)
    
Solve the value function using value function iteration.
Returns the value function and optimal order policy.
"""
function solve_value_function(params; tol=1e-4, maxiter=1000, full=false, fast_interp=true,verbose=false)
    Sgrid = params.Sgrid
    Ns = params.Ns
    ω_grid = params.ω_grid
    P_ω = params.P_ω
    Nω = params.Q_ω

    # V_by_omega[i, j] = V(s_i, ω_j).  Both inventory and ω are state variables.
    V_by_omega     = zeros(Ns, Nω)
    V_by_omega_new = similar(V_by_omega)

    # Policy matrices (Ns × Nω)
    n_policy_current = zeros(Ns, Nω)
    c_tilde = params.c
    p_policy_current = zeros(Ns, Nω)

    diff = Inf
    iter = 0

    # Initial price guess via static FOC for each ω state
    for j in 1:Nω
        p_policy_current[:, j] .= solve_price_policy(params, c_tilde, ω_grid[j])
    end

    # Precompute demand and operating-cost tables to avoid pow_body in the quadrature loop
    D_table, C_table = precompute_demand(p_policy_current, params)

    # Precompute uniform-grid constants (only used when fast_interp=true)
    inv_h     = (Ns - 1) / (Sgrid[end] - Sgrid[1])
    EV_cont_j = Vector{Float64}(undef, Ns)   # preallocate; reused every (iter, j)

    while diff > tol && iter < maxiter

        for j in 1:Nω
            # State-conditional continuation: E[V(s', ω') | ω_j] for each next-period s'
            if fast_interp
                mul!(EV_cont_j, V_by_omega, P_ω[j, :])
                Vinterp_j = UniformInterp(copy(EV_cont_j), Sgrid[1], inv_h)
            else
                Vinterp_j = LinearInterpolation(Sgrid, V_by_omega * P_ω[j, :], extrapolation_bc=Line())
            end

            n_upper = maximum(Sgrid)
            for i in 1:Ns
                n_t, v = maximize_expected_value_choice(i, j, D_table, C_table, p_policy_current, ω_grid[j], Vinterp_j, params, n_upper=n_upper)
                V_by_omega_new[i, j] = v
                n_policy_current[i, j] = n_t
                if n_t > 0.0
                    n_upper = n_t
                end
            end
        end

        diff = maximum(abs.(V_by_omega_new .- V_by_omega))
        V_by_omega .= V_by_omega_new
        iter += 1
    end

    if verbose
        println("Initial Value Function Solved at $iter iterations")
    end

    if full
        p_lower = params.c * 0.5
        p_upper = maximum(p_policy_current) * 2.0
        diff = Inf
        iter = 0
        maxiter = 500
        n_policy_prev = copy(n_policy_current)
        p_policy_prev = copy(p_policy_current)

        while diff > tol && iter < maxiter
            for j in 1:Nω
                EV_cont_j = V_by_omega * P_ω[j, :]
                Vinterp_j = LinearInterpolation(Sgrid, EV_cont_j, extrapolation_bc=Line())

                n_upper_curr = maximum(Sgrid)
                p_upper_curr = p_upper
                for i in 1:Ns
                    n_t, p_t, v = maximize_expected_value_choice(i, ω_grid[j], Vinterp_j, params,
                        n0=n_policy_prev[i, j], p0=p_policy_prev[i, j],
                        p_lower=p_lower, p_upper=p_upper_curr, n_upper=n_upper_curr)
                    V_by_omega_new[i, j] = v
                    n_policy_current[i, j] = n_t
                    p_policy_current[i, j] = p_t
                    if n_t > 0.0
                        n_upper_curr = max(n_t * 1.01,params.Sgrid[2])
                    end
                    p_upper_curr = p_t * 1.01
                end
            end

            diff = max(maximum(abs.(n_policy_current .- n_policy_prev)),
                       maximum(abs.(p_policy_current .- p_policy_prev)),
                       maximum(abs.(V_by_omega_new .- V_by_omega)))

            n_policy_prev .= n_policy_current
            p_policy_prev .= p_policy_current
            V_by_omega .= V_by_omega_new
            iter += 1
            if verbose
                println("At Iteration $iter, Error: $diff")
            end
        end
        if verbose
            println("Full Value Function Solved at $iter iterations")
        end
    end

    # Ex-ante value: average over ergodic ω distribution
    V = V_by_omega * params.π_ω
    converged = diff <= tol
    return V, n_policy_current, p_policy_current, V_by_omega, converged
end



"""
    truncated_lognormal_mean(νbar, params)
    
Compute the truncated mean of the lognormal distribution.
"""
function truncated_lognormal_mean(νbar, params)
    μ_log = params.dist.μ   # log-space mean of ν
    σ_log = params.dist.σ   # log-space std of ν

    z1 = (log(νbar) - μ_log - σ_log^2) / σ_log
    z0 = (log(νbar) - μ_log) / σ_log

    return exp(μ_log + 0.5 * σ_log^2) * cdf(Normal(), z1) / cdf(Normal(), z0)
end


"""
    stockout_probability(s, p, params)
    
Compute the stockout probability given inventory s and price p.
"""
function stockout_probability(s, p, params)
    νbar = s * p^params.ϵ
    return 1.0 - cdf(params.dist, νbar)
end

"""
    expected_demand(s, p, params)

Compute expected demand (matches SolveModel.jl)
"""
function expected_demand(s, p, params)

    νbar = s * p^params.ϵ

    # Probabilities
    Fbar = cdf(params.dist, νbar)

    Eν_trunc = truncated_lognormal_mean(νbar, params)

    # Expected demand
    return p^(-params.ϵ) * Eν_trunc * Fbar +
           s * (1 - Fbar)
end


### Total Operating Expenses

function operating_expense(ω, d,params)
    return ω*d^(params.γ)
end


"""
    precompute_demand(p_policy, params) -> Array{Float64, 3}

Return a (Q × Q_ω × Ns) array `D_table` where

    D_table[q, j, i] = min(ν_q · p_policy[i,j]^{-ϵ}, Sgrid[i])

and a matching `C_table` where

    C_table[q, j, i] = ω_j · D_table[q,j,i]^γ

with
- q  ∈ 1:Q    — Gauss-Hermite quadrature node index (demand shock ν)
- j  ∈ 1:Q_ω  — Tauchen ω grid index (cost shock)
- i  ∈ 1:Ns   — beginning-of-period inventory grid index

Precomputing both tables eliminates all `pow_body` calls (ν·p^{-ϵ} and D^γ)
from the inner quadrature loop of `expected_value_choice`.
"""
function precompute_demand(p_policy::Matrix{Float64}, params::Parameters)
    Q       = params.Q
    Nω      = params.Q_ω
    Ns      = params.Ns
    ν_nodes = params.quad_nodes_lognormal
    Sgrid   = params.Sgrid
    ω_grid  = params.ω_grid
    γ       = params.γ

    D_table = Array{Float64,3}(undef, Q, Nω, Ns)
    C_table = Array{Float64,3}(undef, Q, Nω, Ns)

    for i in 1:Ns
        s = Sgrid[i]
        for j in 1:Nω
            p       = p_policy[i, j]
            p_neg_ε = p^(-params.ϵ)         # one pow_body per (i,j), not per q
            ω       = ω_grid[j]
            for q in 1:Q
                D = min(ν_nodes[q] * p_neg_ε, s)
                D_table[q, j, i] = D
                C_table[q, j, i] = ω * D^γ  # one pow_body per (q,j,i)
            end
        end
    end

    return D_table, C_table
end

"""
    shock_specific_value(n, s_i, Sgrid, p_policy, ν, Vinterp, params)
    
Compute the firm value for a specific demand shock.
"""
function shock_specific_value(n::Float64, s_i::Int, p_policy::AbstractVector{Float64}, ν::Float64, ω::Float64, Vinterp, params::Parameters)::Float64
    Sgrid = params.Sgrid
    p = p_policy[s_i]
    s = Sgrid[s_i]
    D = min(ν * p^(-params.ϵ), s)
    s_tilde = s - D + n
    opp_cost = ω * D^(params.γ)

    order_cost = if n > 0
        params.fc + params.c * n
    else
        0.0
    end

    return p * D - opp_cost - order_cost + params.β * Vinterp((1 - params.δ) * s_tilde)
end


"""
    expected_value_choice(n, s_i, Sgrid, p_policy, Vinterp, params)
    
Compute the expected value of choosing order quantity n.
"""
function expected_value_choice(n::Float64, s_i::Int, p_policy::AbstractVector{Float64}, ω::Float64, Vinterp, params::Parameters)::Float64
    w = params.quad_weights
    ν_nodes = params.quad_nodes_lognormal

    EV = 0.0

    for i in 1:params.Q
        EV += w[i] * shock_specific_value(n, s_i, p_policy, ν_nodes[i], ω, Vinterp, params)
    end

    return EV / sqrt(pi)
end


#### Payoffs depending on both n and p

function shock_specific_value(n::Float64, p::Float64, s_i::Int, ν::Float64, ω::Float64, Vinterp, params::Parameters)::Float64
    Sgrid = params.Sgrid
    s = Sgrid[s_i]
    D = min(ν * p^(-params.ϵ), s)
    s_tilde = s - D + n
    opp_cost = ω * D^(params.γ)

    order_cost = if n > 0
        params.fc + params.c * n
    else
        0.0
    end

    return p * D - opp_cost - order_cost + params.β * Vinterp((1 - params.δ) * s_tilde)
end


function expected_value_choice(n::Float64, p::Float64, s_i::Int, ω::Float64, Vinterp, params::Parameters)::Float64
    w = params.quad_weights
    ν_nodes = params.quad_nodes_lognormal

    EV = 0.0

    for i in 1:params.Q
        EV += w[i] * shock_specific_value(n, p, s_i, ν_nodes[i], ω, Vinterp, params)
    end

    return EV / sqrt(pi)
end



"""
    maximize_expected_value_choice(s_i, p_policy, Vinterp, params; n0=nothing)
    
Maximize the expected value over the order quantity n.
"""
function maximize_expected_value_choice(s_i::Int, p_policy::AbstractVector{Float64}, ω::Float64, Vinterp, params::Parameters; n0::Union{Nothing, Float64}=nothing, n_upper::Float64=maximum(params.Sgrid))
    Sgrid = params.Sgrid
    function obj(n)
        value = expected_value_choice(n, s_i, p_policy, ω, Vinterp, params)
        return -value
    end
    
    if isnothing(n0)
        n0 = 5.0
    end
    
    lower = 0.0
    upper = n_upper
    
    result = Optim.optimize(obj, lower, upper, Brent(), rel_tol=1e-12, abs_tol=1e-12)
    
    n_opt = result.minimizer
    value_max = -result.minimum

    no_order_value = expected_value_choice(0.0, s_i, p_policy, ω, Vinterp, params)
    if value_max < no_order_value 
        n_opt = 0.0
        value_max = no_order_value
    end
    
    return n_opt, value_max
end


"""
    shock_specific_value(n, D, s_i, p, ω, Vinterp, params)

Variant that accepts a pre-looked-up demand `D` and price `p`, avoiding repeated
`pow_body` calls inside the quadrature loop.
"""
@inline function shock_specific_value_precomp(n::Float64, D::Float64, C::Float64, p::Float64, s_i::Int, Vinterp, params::Parameters)::Float64
    s          = params.Sgrid[s_i]
    s_tilde    = s - D + n
    order_cost = n > 0 ? params.fc + params.c * n : 0.0
    return p * D - C - order_cost + params.β * Vinterp((1 - params.δ) * s_tilde)
end

"""
    expected_value_choice(n, s_i, j, D_col, p, ω, Vinterp, params)

Variant that takes a pre-computed demand column `D_col = view(D_table, :, j, i)`
(length Q) and the corresponding price `p = p_policy[s_i, j]`.
"""
function expected_value_choice(n::Float64, s_i::Int,
                                D_col::AbstractVector{Float64},
                                C_col::AbstractVector{Float64},
                                p::Float64, Vinterp,
                                params::Parameters)::Float64
    w  = params.quad_weights
    EV = 0.0
    for q in 1:params.Q
        EV += w[q] * shock_specific_value_precomp(n, D_col[q], C_col[q], p, s_i, Vinterp, params)
    end
    return EV / sqrt(pi)
end

"""
    maximize_expected_value_choice(s_i, j, D_table, p_policy, ω, Vinterp, params; ...)

Variant that uses the pre-computed demand table to avoid `pow_body` inside the
quadrature loop.  Only used in the initial (non-full) value-function iteration.
"""
function maximize_expected_value_choice(s_i::Int, j::Int,
                                         D_table::Array{Float64,3},
                                         C_table::Array{Float64,3},
                                         p_policy::Matrix{Float64},
                                         ω::Float64, Vinterp, params::Parameters;
                                         n_upper::Float64=maximum(params.Sgrid))
    D_col = view(D_table, :, j, s_i)
    C_col = view(C_table, :, j, s_i)
    p     = p_policy[s_i, j]
    function obj(n)
        return -expected_value_choice(n, s_i, D_col, C_col, p, Vinterp, params)
    end
    result    = Optim.optimize(obj, 0.0, n_upper, Brent(), rel_tol=1e-12, abs_tol=1e-12)
    n_opt     = result.minimizer
    value_max = -result.minimum
    no_order  = expected_value_choice(0.0, s_i, D_col, C_col, p, Vinterp, params)
    if value_max < no_order
        n_opt     = 0.0
        value_max = no_order
    end
    return n_opt, value_max
end


"""
    maximize_expected_value_choice(s_i, Vinterp, params; ...)

Simultaneously finds the optimal price `p` and order `n` using `Fminbox(NelderMead())`
for a given ω state value.
"""
function maximize_expected_value_choice(s_i::Int, ω::Float64, Vinterp, params::Parameters; n0::Union{Nothing, Float64}=nothing, p0::Union{Nothing, Float64}=nothing,
                                             p_lower::Float64=1e-3, p_upper::Float64=50.0, n_upper::Float64=maximum(params.Sgrid))
    lower = [0.0,   p_lower]
    upper = [n_upper, p_upper]
    if !isnothing(p0) && (p0 > p_upper)
        p0 = p_lower + (p_upper - p_lower) / 2
    end
    if !isnothing(n0) && (n0 > n_upper)
        n0 = n_upper / 2
    end
    x0 = [clamp(isnothing(n0) ? 5.0 : n0, 0.0, n_upper),
          clamp(isnothing(p0) ? params.c * params.ϵ / (params.ϵ - 1) : p0, p_lower, p_upper)]

    obj(x) = -expected_value_choice(x[1], x[2], s_i, ω, Vinterp, params)

    local result
    try
        result = Optim.optimize(obj, lower, upper, x0, Fminbox(NelderMead()))
    catch e
        println("ERROR in maximize_expected_value_choice at s_i=$s_i (s=$(params.Sgrid[s_i])), ω=$ω")
        println("  lower = [n=$(lower[1]), p=$(lower[2])]")
        println("  upper = [n=$(upper[1]), p=$(upper[2])]")
        println("  x0    = [n=$(x0[1]), p=$(x0[2])]")
        rethrow(e)
    end
    n_opt     = result.minimizer[1]
    p_opt     = result.minimizer[2]
    value_max = -result.minimum

    if !Optim.converged(result)
        println("WARNING: optimizer did not converge at grid index s_i=$s_i (s=$(params.Sgrid[s_i]))")
        println("  x0        = [n=$(x0[1]), p=$(x0[2])]")
        println("  bounds    = n∈[$(lower[1]), $(upper[1])], p∈[$(lower[2]), $(upper[2])]")
        println("  minimizer = [n=$(round(n_opt, digits=4)), p=$(round(p_opt, digits=4))]")
        println("  minimum   = $(result.minimum)  (converged=$(Optim.converged(result)), iters=$(Optim.iterations(result)))")
    end

    return n_opt, p_opt, value_max
end



# ---------------------------------------------------
# Markov chain helpers for ω draws
# ---------------------------------------------------
"""
    draw_ω_index(params, current_idx)

Draw the next ω grid index from row `current_idx` of the Tauchen transition matrix.
"""
function draw_ω_index(rng::AbstractRNG, params::Parameters, current_idx::Int)
    r = rand(rng)
    cumprob = 0.0
    for k in 1:params.Q_ω
        cumprob += params.P_ω[current_idx, k]
        if r ≤ cumprob
            return k
        end
    end
    return params.Q_ω
end

function draw_ω_index(params::Parameters, current_idx::Int)
    return draw_ω_index(Random.default_rng(), params, current_idx)
end

"""
    draw_ω_index_ergodic(params)

Draw an initial ω grid index from the ergodic stationary distribution `π_ω`.
"""
function draw_ω_index_ergodic(rng::AbstractRNG, params::Parameters)
    r = rand(rng)
    cumprob = 0.0
    for k in 1:params.Q_ω
        cumprob += params.π_ω[k]
        if r ≤ cumprob
            return k
        end
    end
    return params.Q_ω
end

function draw_ω_index_ergodic(params::Parameters)
    return draw_ω_index_ergodic(Random.default_rng(), params)
end


# ---------------------------------------------------
# Simulation of firm dynamics
# ---------------------------------------------------
function simulate_firm(rng::AbstractRNG, num_simulations::Int, num_periods::Int, price_policy_interp, order_policy_interp, params; burn_in::Int=100)
    """
    Simulate firm inventory dynamics over multiple periods and simulations.
    Returns vectors of beginning-of-period inventory, demand, operating expense,
    and revenue.
    """
    Sgrid = params.Sgrid
    all_inventory_levels = Float64[]
    all_demand_levels = Float64[]
    all_expenses = Float64[]
    all_revenue = Float64[]
    
    for sim in 1:num_simulations
        # Random starting inventory; initial ω drawn from ergodic distribution
        s_current = rand(rng, Sgrid)
        ω_idx     = draw_ω_index_ergodic(rng, params)

        for period in 1:burn_in
            p_opt     = price_policy_interp(s_current, ω_idx)
            n_opt     = order_policy_interp(s_current, ω_idx)
            ν         = rand(rng, params.dist)
            D         = min(ν * p_opt^(-params.ϵ), s_current)
            s_current = max((1 - params.δ) * (s_current - D + n_opt), 0.0)
            ω_idx     = draw_ω_index(rng, params, ω_idx)
        end

        for period in 1:num_periods
            # Record beginning-of-period inventory
            push!(all_inventory_levels, s_current)

            # Get optimal price and order quantity using interpolation
            p_opt = price_policy_interp(s_current, ω_idx)

            # Find closest grid point for order policy
            n_opt = order_policy_interp(s_current, ω_idx)

            ω_current = params.ω_grid[ω_idx]

            # Draw demand shock from lognormal
            ν = rand(rng, params.dist)

            # Calculate demand
            D = min(ν * p_opt^(-params.ϵ), s_current)
            push!(all_demand_levels, D)

            # Calculate operating expenses
            c_opp = operating_expense(ω_current,D,params)
            push!(all_expenses,c_opp)

            # Revenue
            push!(all_revenue, p_opt * D)

            # Ending inventory after demand
            s_end = s_current - D

            # Next period beginning inventory: order + depreciation
            s_current = (1 - params.δ) * (s_end + n_opt)

            # Transition ω for next period via Markov chain
            ω_idx = draw_ω_index(rng, params, ω_idx)
        end
    end
    
    return all_inventory_levels, all_demand_levels, all_expenses, all_revenue
end

function simulate_firm(num_simulations::Int, num_periods::Int, price_policy_interp, order_policy_interp, params; burn_in::Int=100)
    return simulate_firm(Random.default_rng(), num_simulations, num_periods, price_policy_interp, order_policy_interp, params; burn_in=burn_in)
end
