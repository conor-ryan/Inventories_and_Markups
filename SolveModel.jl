using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Plots, Interpolations, LineSearch, Random, Statistics, DataFrames, GLM, FixedEffectModels, Printf, BenchmarkTools, Profile
include("ModelFunctions.jl")
include("EstimationFunctions.jl")

params = Parameters(c=1.0, fc=0.0, μη=log(0.1),ση2=0.05,ρ_ω=0.1, γ=0.9,δ=0.05, β=0.95, ϵ=8.0, μν=1, σν2=0.09, Smax=20, Ns=200,scale=1.0,size=100)


# # ---------------------------------------------------
# # Single-iteration benchmark (run interactively before the full solve)
# # ---------------------------------------------------
# println("Benchmarking one value-function iteration...")
# display(@benchmark solve_value_function($params, maxiter=1000,fast_interp=true) samples=10 evals=1)

# # ---------------------------------------------------
# # Profile one iteration and display a flat time profile
# # ---------------------------------------------------
# println("\nProfiling one value-function iteration...")
# Profile.clear()
# @profile solve_value_function(params, maxiter=1000)
# Profile.print(sortedby=:count, mincount=25)
# # ---------------------------------------------------

Sgrid = params.Sgrid
Smax = params.Smax
Ns = params.Ns

# # Solve model using ModelFunctions
# println("Solving price policy...")
# p_policy = solve_price_policy(params,params.c)

# ind = 3:length(params.Sgrid)
# plot(params.Sgrid[ind],p_policy[ind])


println("Solving value function...")
p_policy, order_policy, V, V_by_omega, price_policy_interp, order_policy_interp, Vinterp = solve_model(params);

# p_policy_full, order_policy_full, V_full, V_by_omega_full, price_policy_interp, order_policy_interp, Vinterp = solve_model(params,full=true);


# V_true, order_policy_true, p_policy_true = solve_value_function( params,tol=1e-5,full=true);

# Omega-integrated policy diagnostics using the ergodic distribution of the AR(1) ω process
ω_weights = params.π_ω
ω_nodes    = params.ω_grid

p_policy_integrated    = p_policy    * ω_weights
order_policy_integrated = order_policy * ω_weights

# Omega indices for per-omega slices (Tauchen grid: index 1 = lowest, Q_ω = highest)
Q = params.Q_ω
j_low  = 1
j_med  = (Q + 1) ÷ 2
j_high = Q
ω_slice_labels = ["E_ω" "ω low" "ω med" "ω high"]

stockouts = zeros(Ns)
stockouts_by_omega = zeros(Ns, Q)
value_derivative = zeros(Ns)
for (i, s) in enumerate(Sgrid)
    stockouts[i] = sum(ω_weights[j] * stockout_probability(s, p_policy[i, j], params) for j in eachindex(ω_nodes))
    for j in 1:Q
        stockouts_by_omega[i, j] = stockout_probability(s, p_policy[i, j], params)
    end
    value_derivative[i] = (Vinterp(min(s + 1e-4, Smax)) - Vinterp(max(s - 1e-4, 0.0))) / (2e-4)
end


# Run simulation
println("Running simulation...")
Random.seed!(212311)  # Set seed for reproducibility
_n_periods = 25000   
inventory_sim, demand_shocks_sim, demand_levels_sim, inv_eop_sim, expense_sim, ω_sim, inv_sales_ratio_sim = simulate_firm(40, _n_periods, price_policy_interp, order_policy_interp, params)
println("Simulation complete. $(length(inventory_sim)) observations recorded.")

# Run statistics simulation
println("\nRunning statistics simulation...")
Random.seed!(212311)  # Set seed for reproducibility
stats = compute_firm_statistics(100, 500, price_policy_interp, order_policy_interp, params)
println("Statistics simulation complete.")
println("\n=== Firm Statistics ===")
println("Average Inventory: $(round(stats.avg_inventory, digits=4))")
println("Variance of Inventory: $(round(stats.var_inventory, digits=4))")
println("Average Markup (p/c): $(round(stats.avg_markup, digits=4))")
println("Average Sales: $(round(mean(demand_levels_sim), digits=4))")
println("Variance of Markup: $(round(stats.var_markup, digits=4))")
println("Average Stockout Probability: $(round(stats.avg_stockout, digits=4))")
println("Variance of Stockout Probability: $(round(stats.var_stockout, digits=4))")
println("Correlation (Markup, Inventory): $(round(stats.corr_markup_inventory, digits=4))")
println("Correlation (Markup, Inventory-to-Sales Ratio): $(round(stats.corr_markup_inventory_ratio, digits=4))")
println("Average Inv-to-Sales Ratio (Avg Inv / Revenue): $(round(stats.inv_to_sales_ratio_avg, digits=4))")
println("Variance of Inv-to-Sales Ratio (Avg Inv / Revenue): $(round(stats.inv_to_sales_ratio_var, digits=4))")


# ---------------------------------------------------
# Plotting function
# ---------------------------------------------------

# # Compute transition-based expected derivative object
# transition_matrix = compute_inventory_transition_matrix(params, price_policy_interp, order_policy_interp)
# expected_nu_value_derivative = expected_next_value_derivative_times_nu(
# value_derivative,
# transition_matrix,
# params,
# price_policy_interp,
# order_policy_interp
# )

# Create combined subplot figure
p1 = plot(Sgrid[2:end], [p_policy_integrated[2:end] p_policy[2:end, j_low] p_policy[2:end, j_med] p_policy[2:end, j_high]],
    xlabel="Inventory Level (s)", ylabel="Price", title="Pricing Policy",
    label=ω_slice_labels, linewidth=2, linestyle=[:solid :dash :dot :dashdot])
p2 = plot(Sgrid, [V V_by_omega[:, j_low] V_by_omega[:, j_med] V_by_omega[:, j_high]],
    xlabel="Inventory Level (s)", ylabel="Value", title="Value Function",
    label=ω_slice_labels, linewidth=2, linestyle=[:solid :dash :dot :dashdot])
p3 = plot(Sgrid, [order_policy_integrated order_policy[:, j_low] order_policy[:, j_med] order_policy[:, j_high]],
    xlabel="Inventory Level (s)", ylabel="Orders", title="Order Policy",
    label=ω_slice_labels, linewidth=2, linestyle=[:solid :dash :dot :dashdot])
p4 = histogram(inventory_sim, xlabel="Inventory Level (s)", ylabel="Frequency", title="BOP Inventory Distribution", legend=false, bins=50)
p5 = plot(Sgrid, [stockouts stockouts_by_omega[:, j_low] stockouts_by_omega[:, j_med] stockouts_by_omega[:, j_high]],
    xlabel="Inventory Level (s)", ylabel="Stockout Probability", title="Stockout Probability",
    label=ω_slice_labels, linewidth=2, linestyle=[:solid :dash :dot :dashdot])
p6 = plot(Sgrid, value_derivative, xlabel="Inventory Level (s)", ylabel="dV/dS", title="Value Function Derivative", legend=false, linewidth=2)
hline!(p6, [params.c/(params.β*(1-params.δ))], linewidth=2, linestyle=:dash, color=:red, label="params.c")
isr_qty = filter(x -> isfinite(x) && x > 0, inventory_sim ./ max.(demand_levels_sim, eps()))
p7 = histogram(isr_qty, xlabel="Inventory / Sales (quantity)", ylabel="Frequency", title="Inv-to-Sales Ratio (Quantity)", legend=false, bins=50)
isr_rev = filter(x -> isfinite(x) && x > 0, inv_sales_ratio_sim ./ params.c)
p8 = histogram(isr_rev, xlabel="Inventory / Sales (revenue)", ylabel="Frequency", title="Inv-to-Sales Ratio (Revenue)", legend=false, bins=50)
let
    revenue_sim = params.c .* inventory_sim ./ inv_sales_ratio_sim   # p·D = c·s / (c·s/(p·D))
    valid9      = (inv_sales_ratio_sim .> 0) .& isfinite.(revenue_sim) .& (revenue_sim .> 0)
    isr_bom_rev = filter(isfinite, inventory_sim[valid9] ./ revenue_sim[valid9])
    global p9   = histogram(isr_bom_rev, xlabel="BOM Inventory / Revenue", ylabel="Frequency",
                            title="Inv-to-Sales Ratio (BOM Inv, Revenue)", legend=false, bins=50)
end
# p10 = plot(Sgrid, expected_nu_value_derivative,
# xlabel="Inventory Level (s)",
# ylabel="E[νV'(s')|s] / E[ν|ν≤ν̄(s)]",
# title="Normalized Expected ν·dV/dS at Next State",
# legend=false,
# linewidth=2)

combined_plot = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, layout=(3,3), size=(1600, 1600), margin=5Plots.mm)
display(combined_plot)



# s_trans = compute_inventory_transition_matrix(params,price_policy_interp,order_policy_interp)


# y = dVds_approx(params,p_policy,order_policy)  

# plot(params.Sgrid,y)
# ind = findall((inventory_sim.>13) .& (inventory_sim.<15)) 
# ind_pre = ind .-1
# histogram(demand_shocks_sim[ind_pre])
# histogram(inv_eop_sim[ind])


# # ---------------------------------------------------
# # OLS estimation of γ: log(expenses) = α + γ·log(demand) + ε
# # Theoretical relationship: expense = ω·D^γ  →  log(expense) = log(ω) + γ·log(D)
# # ---------------------------------------------------
# mask = (expense_sim .> 0) .& (demand_levels_sim .> 0) .& (demand_shocks_sim .> 0) .& (ω_sim .> 0)

# # Pre-compute Δ(inv/sales ratio), Δlog(ω), previous-period log(ω), and forward Δ(inv/sales)
# _n_periods = 25000   # must match the simulate_firm call above
# Δinv_sales_full      = similar(inv_sales_ratio_sim)
# Δinv_sales_fwd_full  = similar(inv_sales_ratio_sim)
# Δlog_omega_full      = similar(ω_sim)
# log_omega_prev_full  = similar(ω_sim)
# log_expense_prev_full = similar(expense_sim)
# Δinv_sales_full[1]        = NaN
# Δinv_sales_fwd_full[end]  = NaN
# Δlog_omega_full[1]        = NaN
# log_omega_prev_full[1]    = NaN
# log_expense_prev_full[1]  = NaN
# for t in 2:length(inv_sales_ratio_sim)
#     is_boundary = (t - 1) % _n_periods == 0
#     Δinv_sales_full[t]         = is_boundary ? NaN : inv_sales_ratio_sim[t] - inv_sales_ratio_sim[t - 1]
#     Δinv_sales_fwd_full[t - 1] = is_boundary ? NaN : inv_sales_ratio_sim[t] - inv_sales_ratio_sim[t - 1]
#     Δlog_omega_full[t]          = is_boundary ? NaN : log(ω_sim[t]) - log(ω_sim[t - 1])
#     log_omega_prev_full[t]      = is_boundary ? NaN : log(ω_sim[t - 1])
#     log_expense_prev_full[t]    = (is_boundary || expense_sim[t - 1] <= 0) ? NaN : log(expense_sim[t - 1])
# end

# mask_prev = mask .& .!isnan.(log_omega_prev_full) .& .!isnan.(log_expense_prev_full)
# df_reg = DataFrame(log_expense      = log.(expense_sim[mask_prev]),
#                    log_demand       = log.(demand_levels_sim[mask_prev]),
#                    log_shock        = log.(demand_shocks_sim[mask_prev]),
#                    log_omega        = log.(ω_sim[mask_prev]),
#                    log_omega_prev   = log_omega_prev_full[mask_prev],
#                    log_expense_prev = log_expense_prev_full[mask_prev])

# # firm_boundary_full[t] = true for the first valid row of each firm in df_reg_Δ.
# # Each firm's valid rows start at period 2 of that firm's block: t = (k-1)*_n_periods + 2.
# firm_boundary_full = [(t >= 2) && ((t - 2) % _n_periods == 0) for t in eachindex(inv_sales_ratio_sim)]

# valid_Δ = mask_prev .& .!isnan.(Δinv_sales_full) .& .!isnan.(Δinv_sales_fwd_full)
# df_reg_Δ = DataFrame(log_expense       = log.(expense_sim[valid_Δ]),
#                       log_demand        = log.(demand_levels_sim[valid_Δ]),
#                       log_shock         = log.(demand_shocks_sim[valid_Δ]),
#                       log_omega         = log.(ω_sim[valid_Δ]),
#                       log_omega_prev    = log_omega_prev_full[valid_Δ],
#                       log_expense_prev  = log_expense_prev_full[valid_Δ],
#                       Δinv_sales        = Δinv_sales_full[valid_Δ],
#                       Δinv_sales_fwd    = Δinv_sales_fwd_full[valid_Δ],
#                       Δlog_omega        = Δlog_omega_full[valid_Δ],
#                       firm_boundary     = firm_boundary_full[valid_Δ])

# # 4. OLS: Δ(inv/sales ratio) ~ log(demand shock ν)
# ols_Δ_shock = lm(@formula(log_expense ~ log_demand), df_reg_Δ)
# println("\n=== OLS: Δ(inv/sales) ~ log(demand shock ν) ===")
# display(coeftable(ols_Δ_shock))

# # 5. OLS: Δ(inv/sales ratio) ~ Δlog(cost shock ω)
# ols_Δ_omega = lm(@formula(Δinv_sales ~ log_omega_prev), df_reg_Δ)
# println("\n=== OLS: Δ(inv/sales) ~ Δlog(cost shock ω) ===")
# display(coeftable(ols_Δ_omega))

# # 6. IV: log(expense) ~ log(demand), instrument = log(ν)
# #    First stage: log(D) ~ log(ν); Second stage: log(expense) ~ log(D)̂
# iv = reg(df_reg, @formula(log_expense ~ (log_demand ~ log_shock)))
# println("\n=== IV: log(expense) ~ log(demand), instrument = log(ν) ===")
# display(coeftable(iv))
# resid_iv = FixedEffectModels.residuals(iv,df_reg)
# cov_resid_shock  = cov(resid_iv, df_reg.log_shock)
# println("Cov(residual, log(ν)): $(round(cov_resid_shock, digits=6))")

# # 7. IV: log(expense) ~ log(demand) + log(expense_prev), instrument = Δ(inventory/sales ratio)
# iv2 = reg(df_reg_Δ, @formula(log_expense ~ (log_demand ~ Δinv_sales)))
# println("\n=== IV: log(expense) ~ log(demand), instrument = Δ(inv/sales ratio) ===")
# display(coeftable(iv2))
# resid_iv2 = FixedEffectModels.residuals(iv2,df_reg)
# cov_resid_shock  = cov(resid_iv2, df_reg.log_shock)
# println("Cov(residual, log(ν)): $(round(cov_resid_shock, digits=6))")

# println("\nTrue γ: $(params.γ)")

# # ---------------------------------------------------
# # Bias decomposition for regression 7
# #
# # True model:  log(expense_t) = α + γ·log(D_t) + log(ω_t)
# # Instrument:  z_t = Δ(inv/sales)_t
# #
# # IV plim:  γ̂_IV → γ  +  Cov(z, log ω) / Cov(z, log D)
# #                         ─────────────────────────────
# #                           bias from ω contaminating z
# # ---------------------------------------------------
# z       = df_reg_Δ.Δinv_sales
# log_D   = df_reg_Δ.log_demand
# log_ω   = df_reg_Δ.log_omega
# log_exp = df_reg_Δ.log_expense
# log_ν   = df_reg_Δ.log_shock

# cov_z_D  = cov(z, log_D)
# cov_z_ω  = cov(z, log_ω)
# cov_z_E  = cov(z, log_exp)

# # Oracle bias (uses simulated log ω — not available in real data)
# bias_oracle   = cov_z_ω / cov_z_D
# γ̂_z_IV       = cov_z_E  / cov_z_D   # = plim of reg-7 estimate
# γ̂_ν_IV       = cov(z, log_ν) != 0 ?  # use ν-IV as the consistent benchmark
#                     coef(iv)[end] :    # coefficient on log_demand from reg 6
#                     NaN

# println("\n=== Bias Decomposition: Regression 7 (oracle, uses log ω) ===")
# println("Cov(z, log D)  [first stage]:   $(round(cov_z_D,      digits=6))")
# println("Cov(z, log ω)  [excl. viol.]:   $(round(cov_z_ω,      digits=6))")
# println("Cov(z, log E)  [reduced form]:  $(round(cov_z_E,      digits=6))")
# println("Bias = Cov(z,logω)/Cov(z,logD): $(round(bias_oracle,  digits=6))")
# println("γ̂_z-IV  (= γ + bias):           $(round(γ̂_z_IV,      digits=6))")
# println("True γ:                         $(params.γ)")

# # # ---------------------------------------------------
# # # Iterative bias-corrected estimation
# # ---------------------------------------------------
# γ̂_BC, μω_est, σω2_est, ρω_est = estimate_gamma_bc(params, df_reg_Δ)
# println("True γ:                     $(params.γ)")
# println("True  ω parameters —  μη: $(round(params.μη, digits=6))         ση2: $(round(params.ση2, digits=6))   ρω: $(round(params.ρ_ω, digits=6))")