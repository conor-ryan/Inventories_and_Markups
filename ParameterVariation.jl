using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Interpolations, DataFrames, CSV, Random, Statistics, Plots

include("ModelFunctions.jl")

"""
    generate_random_parameters(δ_grid, ϵ_grid, σν2_grid; seed=nothing, Smax=100, Ns=800)
    
Generate parameter sets for all combinations of the grids.
Returns a vector of Parameter objects.
"""
function generate_random_parameters(δ_grid, ϵ_grid, σν2_grid; c_val=1.0, fc_val=0.0, β_val=0.95, μν_val=100.0, Q=19, seed=nothing, Smax=100, Ns=800)
    # Produce a list of Parameters objects for every combination of the three grids
    if !isnothing(seed)
        Random.seed!(seed)
    end

    param_list = Parameters[]
    for δ_val in δ_grid, ϵ_val in ϵ_grid, σν2_val in σν2_grid
        param = Parameters(
            c=c_val,
            fc=fc_val,
            δ=δ_val,
            β=β_val,
            ϵ=ϵ_val,
            μν=μν_val,
            σν2=σν2_val,
            Q=Q,
            Smax=Smax,
            Ns=Ns
        )
        push!(param_list, param)
    end

    return param_list
end


"""
    run_parameter_variation(; δ_grid=..., ϵ_grid=..., σν2_grid=..., verbose=true, seed=nothing, 
                           num_firms=100, num_periods=500, Smax=100, Ns=800)
    
Run model solution and statistics computation for multiple random parameter sets.
Returns a DataFrame with all results.
"""
function run_parameter_variation(; δ_grid=range(0.05, 0.40, length=5),
                                    ϵ_grid=range(4.0, 8.0, length=5),
                                    σν2_grid=[5e2, 1e3, 1e4],
                                    verbose=true, seed=nothing,
                                    num_firms=100, num_periods=500,
                                    Smax=100, Ns=800)

        # Generate parameter combinations
        if verbose
            println("Generating parameter grid (δ x ϵ x σν2)...")
        end
        param_list = generate_random_parameters(δ_grid, ϵ_grid, σν2_grid; seed=seed, Smax=Smax, Ns=Ns)
    
    # Initialize results storage
    results_data = []
    
    # Loop over each parameter set
    for (idx, params) in enumerate(param_list)
        if verbose
            println("\n=== Parameter Set $idx / $(length(param_list)) ===")
            println("Parameters: c=$(round(params.c, digits=4)), fc=$(round(params.fc, digits=4)), δ=$(round(params.δ, digits=4)), " *
                   "β=$(round(params.β, digits=4)), ϵ=$(round(params.ϵ, digits=4)), " *
                   "μν=$(round(params.μν, digits=4)), σν=$(round(params.σν, digits=4))")
        end
        
        try
            # Solve the model
            if verbose
                println("Solving model...")
            end
            p_policy, order_policy, V, price_policy_interp, order_policy_interp, Vinterp = 
                solve_model(params, verbose=false)
            
            # Compute firm statistics
            if verbose
                println("Computing statistics...")
            end
            stats = compute_firm_statistics(num_firms, num_periods,
                                          price_policy_interp, order_policy_interp, params)
            
            # Store results as a dictionary row
            row = Dict(
                # Parameters
                :c => params.c,
                :fc => params.fc,
                :δ => params.δ,
                :β => params.β,
                :ϵ => params.ϵ,
                :μ => params.μν,
                :σ => params.σν,
                :σν2 => params.σν2_level,
                # Statistics
                :avg_inventory => stats.avg_inventory,
                :var_inventory => stats.var_inventory,
                :avg_markup => stats.avg_markup,
                :var_markup => stats.var_markup,
                :avg_stockout => stats.avg_stockout,
                :var_stockout => stats.var_stockout,
                :corr_markup_inventory => stats.corr_markup_inventory
            )
            
            push!(results_data, row)
            
            if verbose
                println("✓ Completed")
                println("  Avg Inventory: $(round(stats.avg_inventory, digits=4))")
                println("  Avg Markup: $(round(stats.avg_markup, digits=4))")
                println("  Avg Stockout: $(round(stats.avg_stockout, digits=4))")
            end
            
        catch e
            if verbose
                println("✗ Error: $e")
            end
            # Continue to next parameter set even if one fails
        end
    end
    
    # Convert to DataFrame
    results_df = DataFrame(results_data)
    
    return results_df
end


"""
    save_results(results_df, filename)
    
Save results DataFrame to a CSV file.
"""
function save_results(results_df, filename)
    CSV.write(filename, results_df)
    println("Results saved to: $filename")
end


# ============================================================================
# MAIN: Example usage
# ============================================================================

# Set up state grid parameters (will be passed to Parameters constructor)
Smax = 100
Ns = 400

# Run parameter variation over a grid
# Define grids for the three varying parameters
δ_grid = range(0.1, 0.3, length=3)
ϵ_grid = range(4.0, 10.0, length=3)
σν2_grid = [5e2, 1e3, 5e3]

results_df = run_parameter_variation(;
    δ_grid=δ_grid,
    ϵ_grid=ϵ_grid,
    σν2_grid=σν2_grid,
    verbose=true,
    seed=123,
    num_firms=100,
    Smax=Smax,
    Ns=Ns,
    num_periods=500
)

# Display results
println("\n" * "="^80)
println("RESULTS SUMMARY")
println("="^80)
println(results_df)

# Plot parameter variation: 3 rows (δ, ϵ, σν2) × 4 columns (avg inventory, avg markup, corr, avg stockout)
function plot_parameter_variation(results_df::DataFrame, δ_grid, ϵ_grid, σν2_grid; savepath=nothing)
    # helper to fetch stat for specific parameter combo (returns NaN if missing)
    function fetch_stat(col::Symbol, δv, ϵv, σv)
        sub = filter(row -> isapprox(row.δ, δv; atol=1e-12) && isapprox(row.ϵ, ϵv; atol=1e-12) && isapprox(row.σν2, σv; atol=1e-8), results_df)
        if nrow(sub) == 0
            return NaN
        else
            return sub[1, col]
        end
    end

    # medians of the grids (hold constant when varying one parameter)
    med_δ = δ_grid[2]
    med_ϵ = ϵ_grid[2]
    med_σ = σν2_grid[2]

    metrics = (
        :avg_inventory => (ylabel="Average Inventory"),
        :avg_markup => (ylabel="Average Markup"),
        :corr_markup_inventory => (ylabel="Corr(Markup,Inventory)"),
        :var_inventory => (ylabel="Variance of Inventory"),
        :avg_stockout => (ylabel="Average Stockout")
    )

    plt = plot(layout=(3,5), size=(1800,1000), margin=8Plots.mm)

    # Row 1: vary δ
    xs = collect(δ_grid)
    for (j, (col_sym, meta)) in enumerate(metrics)
        ys = Float64[]
        for δv in xs
            push!(ys, fetch_stat(col_sym, δv, med_ϵ, med_σ))
        end
        plot!(plt, xs, ys, subplot=j, xlabel="Depreciation δ", title=meta, legend=false)
    end

    # Row 2: vary ϵ
    xs2 = collect(ϵ_grid)
    for (j, (col_sym, meta)) in enumerate(metrics)
        ys = Float64[]
        for ϵv in xs2
            push!(ys, fetch_stat(col_sym, med_δ, ϵv, med_σ))
        end
        plot!(plt, xs2, ys, subplot=5 + j, xlabel="Elasticity ϵ", legend=false)
    end

    # Row 3: vary σν2 (use log scale on x)
    xs3 = collect(σν2_grid)
    for (j, (col_sym, meta)) in enumerate(metrics)
        ys = Float64[]
        for σv in xs3
            push!(ys, fetch_stat(col_sym, med_δ, med_ϵ, σv))
        end
        plot!(plt, xs3, ys, subplot=10 + j, xlabel="Variance σν2", xscale=:log10, legend=false)
    end

    display(plt)
    if !isnothing(savepath)
        savefig(plt, savepath)
    end
    return plt
end

# # Save results
# results_filename = "parameter_variation_results.csv"
# save_results(results_df, results_filename)

# Plot panel
plot_parameter_variation(results_df, δ_grid, ϵ_grid, σν2_grid)
