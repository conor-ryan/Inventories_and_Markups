using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations

# ---------------------------------------------------
# Parameters struct (matches SolveModel.jl)
# ---------------------------------------------------
struct Parameters
    c::Float64
    fc::Float64
    ω::Float64
    γ::Float64
    δ::Float64
    β::Float64
    ϵ::Float64
    μν::Float64
    σν2::Float64
    σν::Float64
    σν2_level::Float64
    dist::LogNormal
    Q::Int64
    quad_nodes::Vector{Float64}
    quad_weights::Vector{Float64}
    quad_nodes_lognormal::Vector{Float64}
    Smax::Float64
    Ns::Int64
    Sgrid::Vector{Float64}
    
    function Parameters(; c=1.0, fc=0.0, ω=1.0,γ=1.0,  δ=0.2, β=0.95, ϵ=2.0, μν=100, σν2=2832, Q=19, scale=1.0,size=1.0, Smax=50.0, Ns=800)
        x, w = gausshermite(Q)
        
        scale_parameter = scale^(ϵ)
        c = c*scale
        μν = μν*scale_parameter*size 
        σν2 = σν2*scale_parameter^2*size^2
        
        # Compute log-space parameters from mean and variance of v
        σ2 = log(1 + σν2 / μν^2)
        σ  = sqrt(σ2)
        μ  = log(μν) - 0.5 * σ2

        # Transform quadrature nodes to lognormal draws (v-space)
        x_lognormal = [exp(μ + sqrt(2) * σ * x[i]) for i in 1:Q]
        
        # Create state grid
        Sgrid_vec = collect(range(1e-4, Smax, length=Ns))

        new(c, fc, ω, γ, δ, β, ϵ, μ, σ2, σ, σν2, LogNormal(μ, σ), Q, x, w, x_lognormal, Smax, Ns, Sgrid_vec)
    end
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
    
    # # Recalculate log-space parameters with new σν2 and ϵ
    # scale_parameter = (1.0)^(ϵ_new)  # scale is 1.0 by default in our use case
    # μν_scaled = params.μν  # Already stored in log space, but need original for calculation
    # σ2_new = log(1 + σν2_new / (exp(params.μν) + 0.5 * params.σν^2)^2)
    # σ_new = sqrt(σ2_new)
    μ_params = exp(params.μν + 0.5 * params.σν^2) 
    
    # # Transform quadrature nodes to lognormal draws (v-space) with new σ and μ
    # x_lognormal_new = [exp(μ_new + sqrt(2) * σ_new * params.quad_nodes[i]) for i in 1:params.Q]
    
    # Create new Parameters object
    return Parameters(
        c=params.c,
        fc=params.fc,
        ω = params.ω,
        γ = params.γ,
        δ=δ_new,
        β=params.β,
        ϵ=ϵ_new,
        μν=μ_params,  
        σν2=σν2_new,
        Q=params.Q,
        Smax=params.Smax,
        Ns=params.Ns
    )
end


"""
    solve_price_policy(params)
    
Solve the price policy using the residual equation.
Returns the price policy vector.
"""
function solve_price_policy(params::Parameters,c_tilde::Float64)
    Sgrid = params.Sgrid
    Ns = params.Ns
    p_policy = zeros(Ns)
    
    for (i, s) in enumerate(Sgrid)

        obj(p) = price_residual(p, s, c_tilde, params)^2
        result = Optim.optimize(obj, 1e-3, 50.0, Brent(), rel_tol=1e-16, abs_tol=1e-12)
        p_policy[i] = result.minimizer
    end
    
    return p_policy
end


"""
    price_residual(p, s, params)

Compute pricing residual matching SolveModel.jl
"""
function price_residual(p, s, c_tilde,params)
    νbar = s * p^params.ϵ

    Fbar = cdf(params.dist, νbar)
    tail = 1.0 - Fbar

    # avoid numerical issues
    if Fbar < 1e-10
        return 1e6
    end

    Eν = truncated_lognormal_mean(νbar, params)

    opp_mc = params.ω*params.γ*p^(params.ϵ*(1-params.γ))
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
function solve_value_function(params; tol=1e-4, maxiter=1000,full=false)
    Sgrid = params.Sgrid
    Ns = params.Ns
    V = zeros(Ns)
    Vnew = similar(V)
    n_policy_current = zeros(Ns)
    c_tilde = params.c
    # p_policy_current = zeros(Ns)
    
    diff = Inf
    iter = 0

    p_policy_current= solve_price_policy(params,c_tilde)
    
    while diff > tol && iter < maxiter
        Vinterp = LinearInterpolation(Sgrid, V, extrapolation_bc=Line())
        n_upper = maximum(Sgrid)
        for i in 1:Ns
            n_t, v = maximize_expected_value_choice(i, p_policy_current, Vinterp, params, n_upper=n_upper)
            Vnew[i] = v
            n_policy_current[i] = n_t
            if n_t > 0.0
                n_upper = n_t
            end
        end
        
        diff = maximum(abs.(Vnew .- V))
        V .= Vnew
        iter += 1
    end

    println("Initial Value Function Solved at $iter iterations")
    if full 
        p_lower = params.c*0.5
        p_upper = maximum(p_policy_current)*2
        diff = Inf
        iter = 0
        maxiter = 500
        n_policy_prev = copy(n_policy_current)
        p_policy_prev = copy(p_policy_current)
        while diff > tol && iter < maxiter
            Vinterp = LinearInterpolation(Sgrid, V, extrapolation_bc=Line())
            n_upper_curr = maximum(Sgrid)
            p_upper_curr = p_upper
            for i in 1:Ns
                n_t, p_t, v = maximize_expected_value_choice(i, Vinterp, params, n0=n_policy_prev[i], p0=p_policy_prev[i], p_lower=p_lower, p_upper=p_upper_curr, n_upper=n_upper_curr)
                Vnew[i] = v
                n_policy_current[i] = n_t
                p_policy_current[i] = p_t
                if n_t > 0.0
                    n_upper_curr = n_t *1.01
                end
                p_upper_curr = p_t *1.01
            end
            
            diff = max(maximum(abs.(n_policy_current .- n_policy_prev)),
                    maximum(abs.(p_policy_current .- p_policy_prev)),
                    maximum(abs.(Vnew .- V)))
            # diff = maximum(abs.(Vnew .- V))
            n_policy_prev .= n_policy_current
            p_policy_prev .= p_policy_current
            V .= Vnew
            iter += 1
            println("At Iteration $iter, Error: $diff")
        end

        println("Full Value Function Solved at $iter iterations")
    end
    return V, n_policy_current, p_policy_current
end



"""
    truncated_lognormal_mean(νbar, params)
    
Compute the truncated mean of the lognormal distribution.
"""
function truncated_lognormal_mean(νbar, params)
    # Use stored log-space parameters (matching SolveModel.jl)
    σ2 = params.σν2
    σ  = params.σν
    μ  = params.μν

    z1 = (log(νbar) - μ - σ2) / σ
    z0 = (log(νbar) - μ) / σ

    return exp(μ + 0.5 * σ2) * cdf(Normal(), z1) / cdf(Normal(), z0)
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
    return p^(-ϵ) * Eν_trunc * Fbar +
           s * (1 - Fbar)
end




"""
    shock_specific_value(n, s_i, Sgrid, p_policy, ν, Vinterp, params)
    
Compute the firm value for a specific demand shock.
"""
function shock_specific_value(n::Float64, s_i::Int, p_policy::AbstractVector{Float64}, ν::Float64, Vinterp, params::Parameters)::Float64
    Sgrid = params.Sgrid
    p = p_policy[s_i]
    s = Sgrid[s_i]
    D = min(ν * p^(-params.ϵ), s)
    s_tilde = s - D + n
    opp_cost = params.ω * D^(params.γ)

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
function expected_value_choice(n::Float64, s_i::Int, p_policy::AbstractVector{Float64}, Vinterp, params::Parameters)::Float64
    w = params.quad_weights
    ν_nodes = params.quad_nodes_lognormal

    EV = 0.0

    for i in 1:params.Q
        EV += w[i] * shock_specific_value(n, s_i, p_policy, ν_nodes[i], Vinterp, params)
    end

    return EV / sqrt(pi)
end


#### Payoffs depending on both n and p

function shock_specific_value(n::Float64, p::Float64, s_i::Int, ν::Float64, Vinterp, params::Parameters)::Float64
    Sgrid = params.Sgrid
    s = Sgrid[s_i]
    D = min(ν * p^(-params.ϵ), s)
    s_tilde = s - D + n
    opp_cost = params.ω * D^(params.γ)

    order_cost = if n > 0
        params.fc + params.c * n
    else
        0.0
    end

    return p * D - opp_cost - order_cost + params.β * Vinterp((1 - params.δ) * s_tilde)
end


function expected_value_choice(n::Float64, p::Float64, s_i::Int, Vinterp, params::Parameters)::Float64
    w = params.quad_weights
    ν_nodes = params.quad_nodes_lognormal

    EV = 0.0

    for i in 1:params.Q
        EV += w[i] * shock_specific_value(n, p, s_i, ν_nodes[i], Vinterp, params)
    end

    return EV / sqrt(pi)
end



"""
    maximize_expected_value_choice(s_i, p_policy, Vinterp, params; n0=nothing)
    
Maximize the expected value over the order quantity n.
"""
function maximize_expected_value_choice(s_i::Int, p_policy::AbstractVector{Float64}, Vinterp, params::Parameters; n0::Union{Nothing, Float64}=nothing, n_upper::Float64=maximum(params.Sgrid))
    Sgrid = params.Sgrid
    function obj(n)
        value = expected_value_choice(n, s_i, p_policy, Vinterp, params)
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

    no_order_value = expected_value_choice(0.0, s_i, p_policy, Vinterp, params)
    if value_max < no_order_value 
        n_opt = 0.0
        value_max = no_order_value
    end
    
    return n_opt, value_max
end


"""
    maximize_expected_value_choice(s_i, Vinterp, params; n0=nothing, p0=nothing, p_lower=1e-3, p_upper=50.0, n_upper=Smax)

Simultaneously finds the optimal price `p` and order `n` using `Fminbox(LBFGS())`.
The box `[0, n_upper] × [p_lower, p_upper]` enforces weakly decreasing policies:
the caller passes the optimum from the previous (lower-inventory) grid point as the upper bound.
Uses the `(n, p, s_i, ν, Vinterp, params)` payoff definitions.
"""
function maximize_expected_value_choice(s_i::Int, Vinterp, params::Parameters; n0::Union{Nothing, Float64}=nothing, p0::Union{Nothing, Float64}=nothing,
                                             p_lower::Float64=1e-3, p_upper::Float64=50.0, n_upper::Float64=maximum(params.Sgrid))
    lower = [0.0,    0.001]
    upper = [n_upper, p_upper]
    if !isnothing(p0) & (p0>p_upper)
        p0 = 0.001 + (p_upper-0.001)/2
    end
    if !isnothing(n0) & (n0>n_upper)
        n0 =  (n_upper)/2
    end
    x0    = [clamp(isnothing(n0) ? 5.0 : n0, 0.0,    n_upper),
             clamp(isnothing(p0) ? params.c * params.ϵ / (params.ϵ - 1) : p0, p_lower, p_upper)]

    obj(x) = -expected_value_choice(x[1], x[2], s_i, Vinterp, params)

    result    = Optim.optimize(obj, lower, upper, x0, Fminbox(NelderMead()))
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


"""
    compute_firm_statistics(N::Int, T::Int, price_policy_interp, order_policy_interp, params)
    
Simulate N firms for T periods and compute statistics.
Returns a named tuple with all statistics.
"""
function compute_firm_statistics(N::Int, T::Int, price_policy_interp, order_policy_interp, params)
    Sgrid = params.Sgrid
    all_inventories = Float64[]
    all_markups = Float64[]
    all_stockouts = Float64[]
    all_sales = Float64[]
    all_ratio = Float64[]
    
    for firm in 1:N
        s_current = rand(Sgrid)
        
        for period in 1:T
            p_opt = price_policy_interp(s_current)
            n_opt = order_policy_interp(s_current)
            
            push!(all_inventories, s_current)
            push!(all_markups, p_opt / params.c)
            push!(all_stockouts, stockout_probability(s_current, p_opt, params))
            
            ν = rand(params.dist)
            D = min(ν * p_opt^(-params.ϵ), s_current)
            push!(all_sales, D)
            push!(all_ratio, s_current / max(D, eps()))
            
            s_end = s_current - D
            
            s_current = (1 - params.δ) * (s_end + n_opt)
            s_current = max(s_current, 0.0)
        end
    end
    
    avg_sales = mean(all_sales)
    valid_ratio = all_ratio[isfinite.(all_ratio)]
    inv_to_sales_ratio_avg = mean(valid_ratio)
    inv_to_sales_ratio_var = var(valid_ratio)
    inv_to_sales_log_ratio_var = var(log.(1.0 .+ valid_ratio))
    
    corr_markup_inv = cor(all_inventories, all_markups)
    corr_markup_inv_ratio = cor(all_ratio[all_inventories .> 0], all_markups[all_inventories .> 0])
    
    return (
        avg_inventory = inv_to_sales_ratio_avg,
        var_inventory = inv_to_sales_log_ratio_var,
        avg_markup = mean(all_markups),
        var_markup = var(all_markups),
        avg_stockout = mean(all_stockouts),
        var_stockout = var(all_stockouts),
        corr_markup_inventory = corr_markup_inv,
        corr_markup_inventory_ratio = corr_markup_inv_ratio,
        avg_sales = avg_sales,
        inv_to_sales_ratio_avg = inv_to_sales_ratio_avg,
        inv_to_sales_ratio_var = inv_to_sales_ratio_var,
        inv_to_sales_log_ratio_var = inv_to_sales_log_ratio_var
    )
end


"""
    solve_model(params; verbose=false)
    
Solve the complete model: price policy, value function, and order policy.
Returns (p_policy, order_policy, V, price_policy_interp, order_policy_interp, Vinterp)
"""
function solve_model(params; verbose=false)
    Sgrid = params.Sgrid
    # if verbose
    #     println("Solving price policy...")
    # end
    # p_policy = solve_price_policy(params,params.c)
    
    if verbose
        println("Solving value function...")
    end
    V, order_policy = solve_value_function(params)
    
    # Create interpolations
    price_policy_interp = LinearInterpolation(Sgrid, p_policy, extrapolation_bc=Line())
    order_policy_interp = LinearInterpolation(Sgrid, order_policy, extrapolation_bc=Line())
    Vinterp = LinearInterpolation(Sgrid, V, extrapolation_bc=Line())
    
    return p_policy, order_policy, V, price_policy_interp, order_policy_interp, Vinterp
end


"""
    compute_inventory_transition_matrix(params::Parameters, price_policy_fn, order_policy_fn)

Build the Markov transition matrix over inventory states on `params.Sgrid`.

Inputs:
- `params`: model parameters
- `price_policy_fn`: callable returning optimal price at inventory `s`
- `order_policy_fn`: callable returning optimal order at inventory `s`

Returns:
- `P::Matrix{Float64}` with size `(Ns, Ns)`, where `P[i, j] = Pr(s_{t+1} = Sgrid[j] | s_t = Sgrid[i])`

Implementation details:
- Demand shock is continuous lognormal (`params.dist`)
- `s_{t+1}` is mapped to the discrete grid using midpoint bins implied by `Sgrid`
- Stockout events induce a point mass at `s_{t+1} = (1-δ) * n(s)`
"""
function compute_inventory_transition_matrix(params::Parameters, price_policy_fn, order_policy_fn)
    Sgrid = params.Sgrid
    Ns = params.Ns
    δ = params.δ
    depreciation_survival = 1.0 - δ

    if depreciation_survival <= 0.0
        error("Need 1 - δ > 0 to construct the transition matrix.")
    end

    # Midpoint bins for mapping continuous next-period inventory to discrete grid states.
    edges = Vector{Float64}(undef, Ns + 1)
    edges[1] = -Inf
    for j in 1:(Ns - 1)
        edges[j + 1] = 0.5 * (Sgrid[j] + Sgrid[j + 1])
    end
    edges[Ns + 1] = Inf

    P = zeros(Float64, Ns, Ns)

    for (i, s) in enumerate(Sgrid)
        p = max(price_policy_fn(s), eps())
        n = max(order_policy_fn(s), 0.0)

        νbar = s * p^params.ϵ
        s_next_max = depreciation_survival * (s + n)
        s_next_min = depreciation_survival * n

        # Continuous (non-stockout) part: ν ∈ [0, νbar]
        for j in 1:Ns
            low_bin = edges[j]
            high_bin = edges[j + 1]

            # Intersect bin with support of continuous next-inventory values [s_next_min, s_next_max]
            low_cont = max(low_bin, s_next_min)
            high_cont = min(high_bin, s_next_max)

            if high_cont > low_cont
                ν_low = max(0.0, (s + n - high_cont / depreciation_survival) * p^params.ϵ)
                ν_high = min(νbar, (s + n - low_cont / depreciation_survival) * p^params.ϵ)

                if ν_high > ν_low
                    P[i, j] += cdf(params.dist, ν_high) - cdf(params.dist, ν_low)
                end
            end

            # Stockout atom: ν > νbar implies D = s and s_{t+1} = (1-δ) * n
            if (low_bin < s_next_min) && (s_next_min <= high_bin)
                P[i, j] += 1.0 - cdf(params.dist, νbar)
            end
        end

        row_sum = sum(P[i, :])
        if row_sum > 0
            P[i, :] ./= row_sum
        else
            nearest_j = argmin(abs.(Sgrid .- s_next_min))
            P[i, nearest_j] = 1.0
        end
    end

    return P
end


# """
#     expected_next_value_derivative_times_nu(
#         derivative_on_grid::AbstractVector{<:Real},
#         transition_matrix::AbstractMatrix{<:Real},
#         params::Parameters,
#         price_policy_fn,
#         order_policy_fn
#     )

# For each current state `s_t = Sgrid[i]`, compute

#     E[ ν_t * V'(s_{t+1}) | s_t = Sgrid[i] ]

# where `V'(Sgrid[j])` is provided by `derivative_on_grid[j]` and the distribution
# of `s_{t+1}` is discretized on `Sgrid` using `transition_matrix`.

# Because ν and next-state bins are not independent, this function uses
# `price_policy_fn` and `order_policy_fn` to recover bin-conditional moments
# `E[ν | s_t=i, s_{t+1} in bin j]` implied by the model's inventory law of motion.
# """
# function expected_next_value_derivative_times_nu(
#     derivative_on_grid::AbstractVector{<:Real},
#     transition_matrix::AbstractMatrix{<:Real},
#     params::Parameters,
#     price_policy_fn,
#     order_policy_fn
# )
#     Ns = params.Ns
#     Sgrid = params.Sgrid
#     δ = params.δ
#     depreciation_survival = 1.0 - δ
#     μ = params.μν
#     σ = params.σν

#     if depreciation_survival <= 0.0
#         error("Need 1 - δ > 0 to compute expected ν-weighted derivatives.")
#     end
#     if length(derivative_on_grid) != Ns
#         error("derivative_on_grid must have length params.Ns = $(Ns).")
#     end
#     if size(transition_matrix, 1) != Ns || size(transition_matrix, 2) != Ns
#         error("transition_matrix must be of size (params.Ns, params.Ns) = ($(Ns), $(Ns)).")
#     end

#     derivative_vec = Float64.(derivative_on_grid)
#     P = Float64.(transition_matrix)

#     # Midpoint bins for mapping continuous next-period inventory to discrete states.
#     edges = Vector{Float64}(undef, Ns + 1)
#     edges[1] = -Inf
#     for j in 1:(Ns - 1)
#         edges[j + 1] = 0.5 * (Sgrid[j] + Sgrid[j + 1])
#     end
#     edges[Ns + 1] = Inf

#     mean_ν = exp(μ + 0.5 * σ^2)

#     # Truncated first moment: E[ν * 1{a < ν ≤ b}] for lognormal ν.
#     function lognormal_first_moment_interval(a::Float64, b::Float64)
#         if !(b > a)
#             return 0.0
#         end

#         cdf_shift(x) = x <= 0.0 ? 0.0 : cdf(Normal(), (log(x) - μ - σ^2) / σ)
#         return mean_ν * (cdf_shift(b) - cdf_shift(a))
#     end

#     expected_values = zeros(Float64, Ns)

#     for (i, s) in enumerate(Sgrid)
#         p = max(price_policy_fn(s), eps())
#         n = max(order_policy_fn(s), 0.0)

#         νbar = s * p^params.ϵ
#         Fbar = cdf(params.dist, νbar)
#         Eν_trunc_state = truncated_lognormal_mean(νbar, params)*Fbar
#         s_next_max = depreciation_survival * (s + n)
#         s_next_min = depreciation_survival * n

#         for j in 1:Ns
#             p_ij = P[i, j]
#             if p_ij <= 0.0
#                 continue
#             end

#             low_bin = edges[j]
#             high_bin = edges[j + 1]

#             # Recover model-implied probability and ν-moment for this (i,j) bin,
#             # restricting to non-stockout shocks ν ≤ νbar.
#             prob_model = 0.0
#             moment_model = 0.0

#             low_cont = max(low_bin, s_next_min)
#             high_cont = min(high_bin, s_next_max)

#             if high_cont > low_cont
#                 ν_low = max(0.0, (s + n - high_cont / depreciation_survival) * p^params.ϵ)
#                 ν_high = min(νbar, (s + n - low_cont / depreciation_survival) * p^params.ϵ)

#                 if ν_high > ν_low
#                     prob_model += cdf(params.dist, ν_high) - cdf(params.dist, ν_low)
#                     moment_model += lognormal_first_moment_interval(ν_low, ν_high)
#                 end
#             end

#             if prob_model > 0.0
#                 expected_values[i] += derivative_vec[j] * moment_model
#             end
#         end

#         if Eν_trunc_state > 0.0
#             expected_values[i] /= Eν_trunc_state
#         else
#             expected_values[i] = 0.0
#         end
#     end

#     return expected_values
# end



function dVds_approx(params,
                    p_policy,
                    order_policy)


    stockouts = [stockout_probability(params.Sgrid[i], p_policy[i], params) for i in 1:length(params.Sgrid)]

    stockout_policy_interp = LinearInterpolation(Sgrid, stockouts, extrapolation_bc=Line())
    p_policy_interp = LinearInterpolation(Sgrid, p_policy, extrapolation_bc=Line())

    dVds_v_mat = Matrix{Float64}(undef,length(params.Sgrid),length(params.quad_nodes_lognormal))
    for I in CartesianIndices(dVds_v_mat)
        i, j = Tuple(I)

        y = params.quad_nodes_lognormal[j]*p_policy[i]^(-params.ϵ)
        q = min(y,params.Sgrid[i])
        s_prime = (1-params.δ)*(params.Sgrid[i] + order_policy[i] - q )


        y = params.quad_nodes_lognormal[j]*p_policy[i]^(-params.ϵ)
        if y<params.Sgrid[i]
            dVds_v_mat[i,j] =params.β*(1-params.δ)*params.quad_nodes_lognormal[j]*( stockout_policy_interp(s_prime)*p_policy_interp(s_prime) + params.c)
        else 
            dVds_v_mat[i,j]=0.0
        end
    end

    exp_derivative = dVds_v_mat*params.quad_weights/sqrt(pi)

    vbar = params.Sgrid.*p_policy.^params.ϵ
    prob = [cdf(params.dist,v) for v in vbar]
    Ev = [truncated_lognormal_mean(v, params) for v in vbar]

    return exp_derivative./(Ev.*prob)
end
