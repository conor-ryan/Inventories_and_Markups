using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations, Random, Statistics, NLopt

include("ModelFunctions.jl")

"""
    simulate_and_extract_moments(params; N=100, T=500, seed=nothing)
    
Solve the model, simulate firms, and extract three key moments:
1. Average markup
2. Average inventory-to-sales ratio
3. Variance of inventory-to-sales ratio

Returns a NamedTuple with fields: avg_markup, avg_inv_to_sales, var_inv_to_sales
"""
function simulate_and_extract_moments(params; N=1000, T=500, seed=nothing,verbose=false)
    # Set seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Solve the model
    p_policy, order_policy, V, _V_by_omega, price_policy_interp, order_policy_interp, Vinterp = 
        solve_model(params, verbose=verbose)
    
    # Simulate firms and compute statistics
    stats = compute_firm_statistics(N, T, price_policy_interp, order_policy_interp, params)
    
    # Extract the three moments
    avg_markup = stats.avg_markup
    avg_inv_to_sales = stats.inv_to_sales_ratio_avg
    var_inv_to_sales = stats.inv_to_sales_log_ratio_var
    
    return (avg_markup=avg_markup, avg_inv_to_sales=avg_inv_to_sales, var_inv_to_sales=var_inv_to_sales)
end


"""
    estimate_parameters_bobyqa(params_base, target_moments; 
                               δ_lb=0.001, δ_ub=0.95,
                               logσν2_lb=5, logσν2_ub=15,
                               ϵ_lb=1.05, ϵ_ub=20.0,
                               N=100, T=500, seed=nothing, 
                               ftol_rel=1e-6, ftol_abs=1e-8, maxtime=Inf,
                               verbose=false)

Estimate parameters (δ, log(σν2), ϵ) using the Nelder-Mead algorithm from Optim.

Takes:
- params_base: A parameter object with fixed parameters
- target_moments: Target moments from simulate_and_extract_moments()
- δ_lb, δ_ub: Lower and upper bounds for depreciation rate
- logσν2_lb, logσν2_ub: Lower and upper bounds for log of demand shock variance
- ϵ_lb, ϵ_ub: Lower and upper bounds for demand elasticity
- N, T: Number of firms and periods for simulation
- seed: Random seed for reproducibility
- ftol_rel: Relative tolerance for objective function
- ftol_abs: Absolute tolerance for objective function
- maxtime: Maximum time in seconds for optimization
- verbose: Print iteration details (iteration number, parameters, and moments)

Returns a NamedTuple with fields:
- δ_est: Estimated depreciation rate
- σν2_est: Estimated demand shock variance (exponentiated from log)
- ϵ_est: Estimated demand elasticity
- obj_value: Objective function value at solution
- result: Optim result object
"""
function estimate_parameters_bobyqa(x0, params_base, target_moments; 
                                    δ_lb=0.001, δ_ub=0.95,
                                    logσν2_lb=5, logσν2_ub=15,
                                    ϵ_lb=1.05, ϵ_ub=20.0,
                                    N=100, T=500, seed=nothing,
                                    ftol_rel=1e-6, ftol_abs=1e-8, maxtime=Inf,
                                    verbose=false)
    
    # Iteration counter for verbose output
    iteration = Ref(0)
    
    # Print header if verbose
    if verbose
        println("\n===== Parameter Estimation via Nelder-Mead =====")
        println("Target moments:")
        println("  avg_markup=$(round(target_moments.avg_markup, digits=4))")
        println("  avg_inv_to_sales=$(round(target_moments.avg_inv_to_sales, digits=4))")
        println("  var_inv_to_sales=$(round(target_moments.var_inv_to_sales, digits=4))")
        println("Parameter bounds:")
        println("  δ ∈ [$δ_lb, $δ_ub]")
        println("  log(σν2) ∈ [$logσν2_lb, $logσν2_ub]")
        println("  ϵ ∈ [$ϵ_lb, $ϵ_ub]")
        println("Starting optimization...")
    end
    
    # # Initial guess (midpoint of bounds in log space for σν2)
    # x0 = [(δ_lb + δ_ub) / 2, 
    #       (logσν2_lb + logσν2_ub) / 2, 
    #       (ϵ_lb + ϵ_ub) / 2]
    
    # Create objective function closure
    function obj(x::Vector{Float64})
        iteration[] += 1
        δ_est = x[1]
        logσν2_est = x[2]
        σν2_est = exp(logσν2_est)  # Convert from log space
        ϵ_est = x[3]
        
        if verbose
            println("\n--- Iteration $(iteration[]) ---")
            println("  Parameters: δ=$(round(δ_est, digits=6)), σν2=$(round(σν2_est, digits=2)), ϵ=$(round(ϵ_est, digits=4))")
        end
        
        try
            # Create new parameter object with estimated parameters
            params_est = update_parameters(params_base, x)
            
            # Simulate and extract moments with estimated parameters
            simulated_moments = simulate_and_extract_moments(params_est; N=N, T=T, seed=seed, verbose=false)
            
            # Compute sum of squared errors
            sse = (simulated_moments.avg_markup - target_moments.avg_markup)^2 +
                  (simulated_moments.avg_inv_to_sales - target_moments.avg_inv_to_sales)^2 +
                  100*(simulated_moments.var_inv_to_sales -target_moments.var_inv_to_sales)^2
            
            if verbose
                println("  Simulated moments:")
                println("    avg_markup=$(round(simulated_moments.avg_markup, digits=4)) (target: $(round(target_moments.avg_markup, digits=4)))") 
                println("    avg_inv_to_sales=$(round(simulated_moments.avg_inv_to_sales, digits=4)) (target: $(round(target_moments.avg_inv_to_sales, digits=4)))") 
                println("    var_inv_to_sales=$(round(simulated_moments.var_inv_to_sales, digits=4)) (target: $(round(target_moments.var_inv_to_sales, digits=4)))") 
                println("  SSE: $(round(sse, digits=6))")
            end
            
            return sse
        catch err
            if verbose
                println("  Model failed to solve. Returning penalty.")
            end
            # Return a large penalty if model fails to solve
            return 1e10
        end
    end
    
    lower = [δ_lb, logσν2_lb, ϵ_lb]
    upper = [δ_ub, logσν2_ub, ϵ_ub]
    options = Optim.Options(
        f_reltol=ftol_rel,
        f_abstol=ftol_abs,
        time_limit=maxtime,
        show_trace=false,
        store_trace=false
    )

    # Starting Point Evaluation 
    if verbose
        println("\n===== Evaluation at Starting Point =====")
        start = obj(x0)
    end
    
    
    # Run bounded Nelder-Mead optimization
    result = Optim.optimize(obj, lower, upper, x0, Optim.Fminbox(Optim.NelderMead()), options)
    obj_value = Optim.minimum(result)
    x_opt = Optim.minimizer(result)
    
    # Print completion message if verbose
    if verbose
        println("\n===== Optimization Complete =====")
        println("Converged: $(Optim.converged(result))")
        println("Final objective value: $(round(obj_value, digits=6))")
    end
    
    # Extract results
    δ_est = x_opt[1]
    logσν2_est = x_opt[2]
    σν2_est = exp(logσν2_est)  # Convert from log space
    ϵ_est = x_opt[3]
    
    # Return results
    return (δ_est=δ_est, σν2_est=σν2_est, ϵ_est=ϵ_est, obj_value=obj_value, result=result)
end



params_target = Parameters(c=1.2, fc=0.0, δ=0.1, β=0.95, ϵ=8.0, μν=100, σν2=exp(7.0), Smax=50, Ns=300)

moments_target = simulate_and_extract_moments(params_target; N=1000, T=500, seed=212422, verbose=true)
moments_target = (avg_markup = 1.3, avg_inv_to_sales = 1.3, var_inv_to_sales = 0.075)  

x0 = [0.1, 7, 8.0]  # Initial guess for (δ, log(σν2), ϵ)

# x0 = [0.15, log(2832)+2, 3.0]  # Initial guess for (δ, log(σν2), ϵ)

test = estimate_parameters_bobyqa(x0,params_target, moments_target; N=100, T=500, seed=212422, verbose=true)

