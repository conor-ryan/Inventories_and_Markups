using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Plots, Interpolations

# ---------------------------------------------------
# Parameters
# ---------------------------------------------------

struct Parameters
    c::Float64
    fc::Float64
    δ::Float64
    β::Float64
    ϵ::Float64
    μν::Float64
    σν2::Float64
    σν::Float64
    dist::LogNormal
    Q::Int64
    quad_nodes::Vector{Float64}
    quad_weights::Vector{Float64}
    quad_nodes_lognormal::Vector{Float64}
    
    function Parameters(; c=0.3, fc=1.0, δ=0.2, β=0.95, ϵ=6.0, μν=1.0, σν2=3.0, σν=sqrt(σν2), Q=19)
        x, w = gausshermite(Q)
        
        # Compute lognormal parameters
        σ2 = log(1 + σν / μν^2)
        σ  = sqrt(σ2)
        μ  = log(μν) - 0.5 * σ2
        
        # Transform quadrature nodes to lognormal draws
        x_lognormal = [exp(μ + sqrt(2) * σ * x[i]) for i in 1:Q]
        
        new(c, fc, δ, β, ϵ, μν, σν2, σν, LogNormal(μν, σν), Q, x, w, x_lognormal)
    end
end

params = Parameters()



# ---------------------------------------------------
# State space
# ---------------------------------------------------
Smax = 50
Ns   = 200
Sgrid = range(1e-4, Smax, length=Ns)

# ---------------------------------------------------
# Demand shock: lognormal
# ---------------------------------------------------
# dist is now included in params.dist

function truncated_lognormal_mean(νbar, params)
    σ2 = log(1 + params.σν / params.μν^2)
    σ  = sqrt(σ2)
    μ  = log(params.μν) - 0.5 * σ2

    z1 = (log(νbar) - μ - σ2) / σ
    z0 = (log(νbar) - μ) / σ

    return exp(μ + 0.5 * σ2) *
           cdf(Normal(), z1) / cdf(Normal(), z0)
end

function stockout_probability(s, p, params)
    νbar = s * p^params.ϵ
    return 1.0 - cdf(params.dist, νbar)
end


# ---------------------------------------------------
# Solve Non-Linear Pricing Equation 
# ---------------------------------------------------
function expected_demand(s, p, params)

    νbar = s * p^params.ϵ

    # Probabilities
    Fbar = cdf(params.dist, νbar)

    Eν_trunc = truncated_lognormal_mean(νbar, params)

    # Expected demand
    return p^(-ϵ) * Eν_trunc * Fbar +
           s * (1 - Fbar)
end

function price_residual(p, s, params)
    νbar = s * p^params.ϵ

    Fbar = cdf(params.dist, νbar)
    tail = 1.0 - Fbar

    # avoid numerical issues
    if Fbar < 1e-10
        return 1e6
    end

    Eν = truncated_lognormal_mean(νbar, params)

    rhs = (params.ϵ / (params.ϵ - 1)) * params.c +
          (1 / (params.ϵ - 1)) *
          s * p^(params.ϵ + 1) *
          (1 / Eν) *
          (tail / Fbar)

    return p - rhs
end

## Price policy as a function of s. 
p_policy = zeros(Ns)

for (i, s) in enumerate(Sgrid)
    p0 = (params.ϵ / (params.ϵ - 1)) * params.c
    
    obj(p) = price_residual(p, s, params)^2
    
    result = optimize(obj, 1e-3, 50.0, Brent())
    p_policy[i] = result.minimizer
end


println("Pricing policy solved using Equation (13).")


### ---------------------------------------------------
# Inventory Ordering Problem - Direct Maximization
# ---------------------------------------------------
V = zeros(Ns)
Vinterp = LinearInterpolation(Sgrid, V, extrapolation_bc=Line())

function optimal_order_direct(s_tilde, Vinterp, params)

    obj(n) = -( -params.c * n + params.β * Vinterp((1 - params.δ) * (s_tilde + n)) )

    res = optimize(obj, 0.0, maximum(Sgrid),Brent())

    value = - Optim.minimum(res)
    non_order_value = params.β * Vinterp((1 - params.δ) * (s_tilde))
    if value-non_order_value>params.fc
        return Optim.minimizer(res), value - params.fc
    else
        return 0.0, non_order_value
    end
end


optimal_order_direct(10, Vinterp,params)


# function V_tilde(s_tilde, Vinterp, params)
#     orders = optimal_order_direct(s_tilde, Vinterp, params)
    
#     # Include fixed cost if ordering occurs
#     if orders > 0
#         return -params.fc - params.c * orders + params.β * Vinterp((1 - params.δ) * (s_tilde + orders))
#     else
#         return params.β * Vinterp((1 - params.δ) * s_tilde)
#     end
# end

function expected_V_tilde(s, p, V_interp, params)

    # Use pre-computed Gauss–Hermite nodes and weights from params
    w = params.quad_weights
    ν_nodes = params.quad_nodes_lognormal

    EV = 0.0

    for i in 1:params.Q
        # Demand and end-of-period inventory
        D = min(ν_nodes[i] * p^(-params.ϵ), s)
        s_tilde = s - D

        n, v_tilde = optimal_order_direct(s_tilde, V_interp, params)

        EV += w[i] * v_tilde
    end

    return EV / sqrt(pi)
end

### -----------------------------------
# Solve Value Function 
## -----------------------------------

# Value function iteration with convergence criterion
Vnew = similar(V)
tol = 1e-6
maxiter = 1000
diff = Inf
iter = 0

while diff > tol && iter < maxiter
    
    Vinterp = LinearInterpolation(Sgrid, V, extrapolation_bc=Line())
    for (i, s) in enumerate(Sgrid)
        Vnew[i] = p_policy[i] * expected_demand(s, p_policy[i], params) + expected_V_tilde(s,p_policy[i],Vinterp,params)
    end
    
    diff = maximum(abs.(Vnew .- V))
    V .= Vnew
    iter += 1
    
    println("Iteration $iter, sup norm = $diff")
end

println("Value function converged in $iter iterations.")

### Best Responses
Vinterp = LinearInterpolation(Sgrid, V, extrapolation_bc=Line())
order_policy = zeros(Ns)
bop_inventory = zeros(Ns)
stockouts = zeros(Ns)
value_derivative = zeros(Ns)
for (i, s) in enumerate(Sgrid)
    n, v_tilde = optimal_order_direct(s, Vinterp, params)
    order_policy[i] = n
    bop_inventory[i] = (1 - params.δ) * (s + order_policy[i])
    stockouts[i] = stockout_probability(s, p_policy[i], params)
    value_derivative[i] = (Vinterp(min(s + 1e-4, Smax)) - Vinterp(max(s - 1e-4, 0.0))) / (2e-4)
end


# ---------------------------------------------------
# Plotting function
# ---------------------------------------------------

# Create combined subplot figure
p1 = plot(Sgrid, p_policy, xlabel="Inventory Level (s)", ylabel="Optimal Price (p)", title="Pricing Policy", legend=false, linewidth=2)
p2 = plot(Sgrid, V, xlabel="Inventory Level (s)", ylabel="Value", title="Value Function", legend=false, linewidth=2)
p3 = plot(Sgrid, order_policy, xlabel="Inventory Level (s)", ylabel="Orders", title="Optimal Orders", legend=false, linewidth=2)
p4 = plot(Sgrid, bop_inventory, xlabel="Inventory Level (s)", ylabel="BOP Inventory", title="Beginning-of-Period Inventory", legend=false, linewidth=2)
p5 = plot(Sgrid, stockouts, xlabel="Inventory Level (s)", ylabel="Stockout Probability", title="Stockout Probability", legend=false, linewidth=2)
p6 = plot(Sgrid, value_derivative, xlabel="Inventory Level (s)", ylabel="dV/dS", title="Value Function Derivative", legend=false, linewidth=2)
hline!(p6, [params.c/(params.β*(1-params.δ))], linewidth=2, linestyle=:dash, color=:red, label="params.c")

combined_plot = plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(1400, 900), margin=5Plots.mm)
display(combined_plot)

