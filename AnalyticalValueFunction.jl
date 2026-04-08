# AnalyticalValueFunction.jl
#
# Investigate whether V(s, ω) has an analytical closed form characterised by
# a small number of parameters.
#
# STEP 1:  Solve the model, plot V(s,ω), compute numerical derivatives,
#          investigate shape properties, and suggest candidate functional forms.

using Distributions, LinearAlgebra, Optim, FastGaussQuadrature,
    Interpolations, Plots, Statistics, Printf, Random
include("ModelFunctions.jl")

# ---------------------------------------------------------------
# 1.  Solve the model
# ---------------------------------------------------------------
params = Parameters(c=1.0, fc=0.0, μη=log(0.1), ση2=0.05, ρ_ω=0.1,
                    γ=0.9, δ=0.05, β=0.95, ϵ=8.0, μν=1, σν2=0.09,
                    Smax=20, Ns=200, scale=1.0, size=100)

println("Solving model...")
p_policy, order_policy, V, V_by_omega, price_policy_interp,
    order_policy_interp, Vinterp = solve_model(params)

Sgrid  = params.Sgrid          # length-Ns inventory grid
ω_grid = params.ω_grid         # length-Nω cost-shock grid
Nω     = params.Q_ω
Ns     = params.Ns

# Colour palette: one colour per ω value, consistent across all plots
ω_colours = palette(:viridis, Nω)

# Skip a handful of near-zero inventory points to avoid boundary artefacts
skip = 5
s_plot   = Sgrid[skip:end]
Ns_plot  = length(s_plot)

# ---------------------------------------------------------------
# 2.  Plot V(s, ω) on its natural scale
# ---------------------------------------------------------------
plt_V = plot(xlabel="Inventory  s", ylabel="V(s, ω)",
             title="Value Function",
             legend=:outerright, size=(700, 450))
for j in 1:Nω
    plot!(plt_V, s_plot, V_by_omega[skip:end, j];
          label=@sprintf("ω = %.3f", ω_grid[j]),
          color=ω_colours[j])
end
display(plt_V)

# ---------------------------------------------------------------
# 3.  Shift so that everything is strictly positive, then produce
#     a log-V vs log-s plot.
#     If  V ≈ A(ω)·sᵅ  then  log V = log A(ω) + α·log s  is linear.
# ---------------------------------------------------------------
V_shift = max(0.0, -minimum(V_by_omega)) + 1.0   # ensures V + V_shift > 0
V_pos   = V_by_omega .+ V_shift

plt_loglog = plot(xlabel="log s", ylabel="log(V + shift)",
                  title="Log–Log Plot  (power-law check in s)",
                  legend=:outerright, size=(700, 450))
for j in 1:Nω
    plot!(plt_loglog, log.(s_plot), log.(V_pos[skip:end, j]);
          label=@sprintf("ω = %.3f", ω_grid[j]),
          color=ω_colours[j])
end
display(plt_loglog)

# ---------------------------------------------------------------
# 4.  Numerical first and second derivatives w.r.t. s (finite differences)
# ---------------------------------------------------------------
ds      = Sgrid[2] - Sgrid[1]           # uniform step
s_inner = Sgrid[2:end-1]                 # grid for d²V

dV  = diff(V_by_omega; dims=1) ./ ds    # (Ns-1) × Nω   – forward difference
d2V = diff(dV;        dims=1) ./ ds    # (Ns-2) × Nω   – second forward difference

s_dV  = (Sgrid[1:end-1] .+ Sgrid[2:end]) ./ 2   # midpoints for dV

# First derivative
skip2 = 5
plt_dV = plot(xlabel="s", ylabel="dV/ds",
              title="First Derivative  dV/ds",
              legend=:outerright, size=(700, 450))
for j in 1:Nω
    plot!(plt_dV, s_dV[skip2:end], dV[skip2:end, j];
          label=@sprintf("ω = %.3f", ω_grid[j]),
          color=ω_colours[j])
end
display(plt_dV)

# Second derivative
plt_d2V = plot(xlabel="s", ylabel="d²V/ds²",
               title="Second Derivative  d²V/ds²",
               legend=:outerright, size=(700, 450))
for j in 1:Nω
    plot!(plt_d2V, s_inner[skip2:end], d2V[skip2:end, j];
          label=@sprintf("ω = %.3f", ω_grid[j]),
          color=ω_colours[j])
end
display(plt_d2V)

# ---------------------------------------------------------------
# 5.  Log–log plot of dV/ds
#     If V = A·sᵅ  then dV/ds = A·α·s^{α-1},
#     so  log(dV/ds) = const + (α-1)·log s  → slope = α - 1
# ---------------------------------------------------------------
dV_pos = dV .+ max(0.0, -minimum(dV)) .+ 1e-8

plt_logdV = plot(xlabel="log s", ylabel="log(dV/ds + shift)",
                 title="Log–Log of First Derivative  (slope ≈ α−1)",
                 legend=:outerright, size=(700, 450))
for j in 1:Nω
    plot!(plt_logdV, log.(s_dV[skip2:end]), log.(dV_pos[skip2:end, j]);
          label=@sprintf("ω = %.3f", ω_grid[j]),
          color=ω_colours[j])
end
display(plt_logdV)

# ---------------------------------------------------------------
# 6.  ω-dependence at fixed s values
#     Check whether V is linear, power-law, or log-linear in ω
# ---------------------------------------------------------------
s_idx = [round(Int, 0.10*Ns), round(Int, 0.25*Ns),
         round(Int, 0.50*Ns), round(Int, 0.75*Ns), round(Int, 0.90*Ns)]

plt_omega      = plot(xlabel="log ω", ylabel="V(s, ω)",
                      title="V vs ω at Fixed s  (level)",
                      legend=:outerright, size=(700, 450))
plt_omega_logV = plot(xlabel="log ω", ylabel="log(V + shift)",
                      title="V vs ω at Fixed s  (log scale)",
                      legend=:outerright, size=(700, 450))
for idx in s_idx
    lbl = @sprintf("s = %.2f", Sgrid[idx])
    plot!(plt_omega,      log.(ω_grid), V_by_omega[idx, :];      label=lbl, marker=:circle)
    plot!(plt_omega_logV, log.(ω_grid), log.(V_pos[idx, :]);     label=lbl, marker=:circle)
end
display(plt_omega)
display(plt_omega_logV)

# ---------------------------------------------------------------
# 7.  Additive separability check
#     If V(s,ω) = f(s) + g(ω) then V(s,ω) − V(s_ref,ω) = f(s) − f(s_ref)
#     should be *independent of ω* (all curves collapse to a single line)
# ---------------------------------------------------------------
s_ref_idx   = round(Int, 0.5 * Ns)
V_diff_add  = V_by_omega .- V_by_omega[s_ref_idx:s_ref_idx, :]   # (Ns) × Nω

plt_add_sep = plot(xlabel="s", ylabel="V(s,ω) − V(s_ref, ω)",
                   title="Additive Separability Check\n(curves should collapse if V = f(s)+g(ω))",
                   legend=:outerright, size=(700, 450))
for j in 1:Nω
    plot!(plt_add_sep, s_plot, V_diff_add[skip:end, j];
          label=@sprintf("ω = %.3f", ω_grid[j]),
          color=ω_colours[j])
end
display(plt_add_sep)

# ---------------------------------------------------------------
# 8.  Multiplicative separability check
#     If V(s,ω) = f(s)·g(ω) then log V is additively separable:
#     log V(s,ω) − log V(s_ref,ω) = log f(s) − log f(s_ref)
#     should be independent of ω
# ---------------------------------------------------------------
log_V_diff = log.(V_pos) .- log.(V_pos[s_ref_idx:s_ref_idx, :])

plt_mult_sep = plot(xlabel="s", ylabel="log V(s,ω) − log V(s_ref, ω)",
                    title="Multiplicative Separability Check\n(curves should collapse if V = f(s)·g(ω))",
                    legend=:outerright, size=(700, 450))
for j in 1:Nω
    plot!(plt_mult_sep, s_plot, log_V_diff[skip:end, j];
          label=@sprintf("ω = %.3f", ω_grid[j]),
          color=ω_colours[j])
end
display(plt_mult_sep)

# ---------------------------------------------------------------
# 9.  OLS power-law exponent estimates for each ω slice
#     Fit  log(V + shift) = a + α·log(s)  by OLS
# ---------------------------------------------------------------
println("\n" * "="^70)
println("POWER-LAW EXPONENT  (fit:  log V = a + α·log s)")
println("="^70)
log_s_vec = log.(s_plot)
X_ols     = hcat(ones(Ns_plot), log_s_vec)
αs        = zeros(Nω)
for j in 1:Nω
    β_ols = X_ols \ log.(V_pos[skip:end, j])
    αs[j] = β_ols[2]
    @printf("  ω = %.4f  →  α = %.4f  (intercept = %.4f)\n",
            ω_grid[j], β_ols[2], β_ols[1])
end
@printf("\n  Mean α across ω: %.4f    Std: %.6f\n", mean(αs), std(αs))

# ---------------------------------------------------------------
# 10.  Curvature summary: sign of d²V/ds²
# ---------------------------------------------------------------
println("\n" * "="^70)
println("CURVATURE SUMMARY  (sign of d²V/ds²)")
println("="^70)
for j in 1:Nω
    frac_neg = mean(d2V[:, j] .< 0)
    @printf("  ω = %.4f :  %.1f%% of grid points have d²V/ds² < 0 (concave)\n",
            ω_grid[j], 100*frac_neg)
end

# ---------------------------------------------------------------
# 11.  ω-slope analysis: fit  V_mean(ω) = a + b·log ω  and  log V_mean = a + b·log ω
# ---------------------------------------------------------------
log_ω_vec = log.(ω_grid)
mean_V_by_ω = [mean(V_by_omega[:, j]) for j in 1:Nω]

println("\n" * "="^70)
println("ω-SLOPE ANALYSIS")
println("="^70)
@printf("  Mean V by ω:\n")
for j in 1:Nω
    @printf("    log(ω) = %+.4f  (ω = %.4f) :  mean V = %+.4f\n",
            log_ω_vec[j], ω_grid[j], mean_V_by_ω[j])
end

# OLS: mean_V ~ a + b·log(ω)
X_ω = hcat(ones(Nω), log_ω_vec)
β_ω = X_ω \ mean_V_by_ω
@printf("\n  Linear-in-log(ω) fit of mean V:  mean V = %.4f + %.4f·log(ω)\n",
        β_ω[1], β_ω[2])

# ---------------------------------------------------------------
# 12.  Print suggestions for analytical functional forms
# ---------------------------------------------------------------
println("\n" * "="^70)
println("CANDIDATE ANALYTICAL FUNCTIONAL FORMS")
println("="^70)
println("""
Based on the plots and OLS estimates above, the following functional forms
are worth investigating in subsequent steps:

  (A) Power law in s, affine in log ω:
        V(s, ω)  =  [a + b·log ω]  ·  sᵅ
      → Two intercept params (a, b) + one exponent α.

  (B) Affine in sᵅ with separate ω intercept:
        V(s, ω)  =  A(ω)  +  B(ω)·sᵅ
      where A(ω) and B(ω) are themselves fit as power-laws or affine in log ω.

  (C) Additively separable with power basis:
        V(s, ω)  =  f(s)  +  g(ω)
      where f(s) is a polynomial or power-law in s and g(ω) is linear in log ω.

  (D) Log-linear:
        log V(s, ω)  =  a  +  α·log s  +  β·log ω
      A pure Cobb-Douglas form; multiplicatively separable.

  (E) Mixed power-log:
        V(s, ω)  =  a  +  b·sᵅ  +  c·log ω  +  d·sᵅ·log ω
      Includes an interaction term between s and ω.

Next step: for each candidate, perform a nonlinear least-squares fit over the
full (s, ω) grid and compare fitted vs. numerical V(s, ω).
""")

# ---------------------------------------------------------------
# 13.  Combined summary figure  (2 × 4 layout)
# ---------------------------------------------------------------
plt_summary = plot(
    plt_V, plt_loglog, plt_dV, plt_d2V,
    plt_logdV, plt_omega, plt_add_sep, plt_mult_sep;
    layout = (4, 2),
    size   = (1200, 1600),
    left_margin  = 8Plots.mm,
    bottom_margin = 6Plots.mm,
    plot_title = "Value Function Properties — Step 1"
)
display(plt_summary)

savefig(plt_summary, joinpath(@__DIR__, "..", "Notes", "AnalyticalVF_step1.png"))
println("\nSaved summary figure → Notes/AnalyticalVF_step1.png")

# ---------------------------------------------------------------
# 14.  Fit alternative analytical forms and compare fit quality
# ---------------------------------------------------------------
println("\n" * "="^70)
println("STEP 2 PREVIEW: FIT CANDIDATE FUNCTIONAL FORMS")
println("="^70)

# Build stacked panel over (s, ω) excluding boundary points used above.
s_stack = repeat(s_plot, Nω)
ω_stack = repeat(ω_grid, inner=Ns_plot)

log_s_stack = log.(s_stack)
log_ω_stack = log.(ω_stack)

y_level = vec(V_by_omega[skip:end, :])
y_log   = log.(vec(V_pos[skip:end, :]))

function fit_ols_model(name::String, X::Matrix{Float64}, y::Vector{Float64}, terms::Vector{String})
    β = X \ y
    yhat = X * β
    resid = y - yhat
    n = length(y)
    k = size(X, 2)
    sse = sum(abs2, resid)
    sst = sum(abs2, y .- mean(y))
    rmse = sqrt(sse / n)
    r2 = 1.0 - sse / sst
    aic = n * log(sse / n) + 2k
    bic = n * log(sse / n) + k * log(n)
    dof = max(n - k, 1)
    sigma2 = sse / dof
    vcov = sigma2 * pinv(X' * X)
    se = sqrt.(diag(vcov))
    tstat = β ./ se
    return (name=name, β=β, yhat=yhat, rmse=rmse, r2=r2, aic=aic, bic=bic, k=k,
            terms=terms, se=se, tstat=tstat)
end

# Log-target models: fit log(V + shift) = f(log s, log ω, ...)
X_log_1 = hcat(ones(length(y_log)), log_s_stack, log_ω_stack)
X_log_2 = hcat(ones(length(y_log)), log_s_stack, log_ω_stack, log_s_stack .* log_ω_stack)
X_log_3 = hcat(ones(length(y_log)), log_s_stack, log_ω_stack,
               log_s_stack .* log_ω_stack, log_s_stack .^ 2, log_ω_stack .^ 2)
X_log_4 = hcat(ones(length(y_log)), s_stack, log_ω_stack, s_stack .* log_ω_stack)
X_log_5 = hcat(ones(length(y_log)), log_s_stack, s_stack, log_ω_stack,
               log_s_stack .* log_ω_stack, s_stack .* log_ω_stack,
               ω_stack, ω_stack .* log_s_stack)

terms_log_1 = ["const", "log s", "log ω"]
terms_log_2 = ["const", "log s", "log ω", "log s·log ω"]
terms_log_3 = ["const", "log s", "log ω", "log s·log ω", "(log s)^2", "(log ω)^2"]
terms_log_4 = ["const", "s", "log ω", "s·log ω"]
terms_log_5 = ["const", "log s", "s", "log ω", "log s·log ω", "s·log ω", "ω", "ω·log s"]

models_log = [
    fit_ols_model("L1: log(V+) ~ 1 + log s + log ω", X_log_1, y_log, terms_log_1),
    fit_ols_model("L2: log(V+) ~ 1 + log s + log ω + log s·log ω", X_log_2, y_log, terms_log_2),
    fit_ols_model("L3: L2 + (log s)^2 + (log ω)^2", X_log_3, y_log, terms_log_3),
    fit_ols_model("L4: log(V+) ~ 1 + s + log ω + s·log ω", X_log_4, y_log, terms_log_4),
    fit_ols_model("L5: log(V+) ~ 1 + log s + s + log ω + interactions + ω + ω·log s", X_log_5, y_log, terms_log_5)
]

# Level-target models: fit V = g(s, log ω, ...)
X_level_1 = hcat(ones(length(y_level)), s_stack, log_s_stack, log_ω_stack, s_stack .* log_ω_stack)
X_level_2 = hcat(ones(length(y_level)), log_s_stack, log_ω_stack, log_s_stack .* log_ω_stack)
X_level_3 = hcat(ones(length(y_level)), s_stack, s_stack .^ 2, log_s_stack, log_ω_stack,
                 s_stack .* log_ω_stack, log_s_stack .* log_ω_stack,
                 ω_stack, ω_stack .* log_s_stack)
X_level_4 = hcat(ones(length(y_level)), log_s_stack, log_ω_stack,
                 log_s_stack .* log_ω_stack, log_s_stack .^ 2)

terms_level_1 = ["const", "s", "log s", "log ω", "s·log ω"]
terms_level_2 = ["const", "log s", "log ω", "log s·log ω"]
terms_level_3 = ["const", "s", "s^2", "log s", "log ω", "s·log ω", "log s·log ω", "ω", "ω·log s"]
terms_level_4 = ["const", "log s", "log ω", "log s·log ω", "(log s)^2"]

models_level = [
    fit_ols_model("V1: V ~ 1 + s + log s + log ω + s·log ω", X_level_1, y_level, terms_level_1),
    fit_ols_model("V2: V ~ 1 + log s + log ω + log s·log ω", X_level_2, y_level, terms_level_2),
    fit_ols_model("V3: V ~ 1 + s + s^2 + log s + log ω + s·log ω + log s·log ω + ω + ω·log s", X_level_3, y_level, terms_level_3),
    fit_ols_model("V4: V ~ 1 + log s + log ω + log s·log ω + (log s)^2", X_level_4, y_level, terms_level_4)
]

function print_coef_table(model, title)
    println("\n" * "-"^70)
    println(title)
    println(model.name)
    println("-"^70)
    println(@sprintf("%-18s %16s %16s %16s", "Term", "Coef", "StdErr", "t-stat"))
    println("-"^70)
    for i in eachindex(model.β)
        println(@sprintf("%-18s %16.8f %16.8f %16.4f",
                         model.terms[i], model.β[i], model.se[i], model.tstat[i]))
    end
end

function print_all_coef_tables(models, title)
    println("\n" * "="^70)
    println(title)
    println("="^70)
    for (idx, m) in enumerate(models)
        print_coef_table(m, @sprintf("MODEL %d COEFFICIENT TABLE", idx))
    end
end

function print_model_table(models, title)
    println("\n" * "-"^70)
    println(title)
    println("-"^70)
    println("Sorted by R² (higher is better)")
    println(@sprintf("%-52s %8s %10s %10s %10s", "Model", "k", "RMSE", "R²", "AIC"))
    println("-"^70)
    for m in sort(models, by=x -> x.r2, rev=true)
        println(@sprintf("%-52s %8d %10.5f %10.5f %10.2f",
                         m.name, m.k, m.rmse, m.r2, m.aic))
    end
end

print_model_table(models_log,   "LOG-TARGET FITS:  log(V + shift)")
print_model_table(models_level, "LEVEL-TARGET FITS:  V")

print_all_coef_tables(models_level, "COEFFICIENT TABLES: ALL LEVEL-TARGET (LINEAR) REGRESSIONS")

best_log = sort(models_log, by=x -> x.r2, rev=true)[1]
v3_idx = findfirst(m -> occursin("V3:", m.name), models_level)
best_level = isnothing(v3_idx) ? sort(models_level, by=x -> x.r2, rev=true)[1] : models_level[v3_idx]
println("\nBest log-target model by R²: " * best_log.name * @sprintf("  (R² = %.6f)", best_log.r2))
println("Level-target model used for final step (forced): " * best_level.name * @sprintf("  (R² = %.6f)", best_level.r2))

# Reconstruct best-fit value functions on original scale
yhat_log_mat = reshape(best_log.yhat, Ns_plot, Nω)
V_hat_best_log = exp.(yhat_log_mat) .- V_shift

yhat_level_mat = reshape(best_level.yhat, Ns_plot, Nω)
V_hat_best_level = yhat_level_mat

print_coef_table(best_log, "COEFFICIENT TABLE: BEST LOG-TARGET MODEL")
print_coef_table(best_level, "COEFFICIENT TABLE: BEST LEVEL-TARGET MODEL")

# Plot best-fit vs numerical V(s,ω) for representative ω values.
ω_compare_idx = unique([1, round(Int, Nω / 2), Nω])

plt_best_fit_log = plot(xlabel="s", ylabel="V(s, ω)",
                title="Best Log-Target Fit vs Numerical V(s, ω)",
                legend=:outerright, size=(760, 500))
for j in ω_compare_idx
    plot!(plt_best_fit_log, s_plot, V_by_omega[skip:end, j];
          color=ω_colours[j], linewidth=2,
          label=@sprintf("True   ω=%.3f", ω_grid[j]))
    plot!(plt_best_fit_log, s_plot, V_hat_best_log[:, j];
          color=ω_colours[j], linewidth=2, linestyle=:dash,
          label=@sprintf("Fitted ω=%.3f", ω_grid[j]))
end

plt_best_fit_level = plot(xlabel="s", ylabel="V(s, ω)",
                  title="Best Level-Target Fit vs Numerical V(s, ω)",
                  legend=:outerright, size=(760, 500))
for j in ω_compare_idx
    plot!(plt_best_fit_level, s_plot, V_by_omega[skip:end, j];
        color=ω_colours[j], linewidth=2,
        label=@sprintf("True   ω=%.3f", ω_grid[j]))
    plot!(plt_best_fit_level, s_plot, V_hat_best_level[:, j];
        color=ω_colours[j], linewidth=2, linestyle=:dash,
        label=@sprintf("Fitted ω=%.3f", ω_grid[j]))
end

plt_best_fit_two_panel = plot(
    plt_best_fit_log, plt_best_fit_level;
    layout=(1, 2),
    size=(1500, 560),
    plot_title="Best Fits: Log-Target vs Level-Target"
)
display(plt_best_fit_two_panel)

savefig(plt_best_fit_two_panel, joinpath(@__DIR__, "..", "Notes", "AnalyticalVF_best_fit_compare.png"))
println("Saved two-panel fit-comparison figure → Notes/AnalyticalVF_best_fit_compare.png")

# ---------------------------------------------------------------
# 15.  FINAL STEP: Map structural parameters -> V3 coefficients
#
# We treat V3 as the maintained specification:
#   V(s,ω) = β0 + β1 s + β2 s^2 + β3 log s + β4 log ω
#            + β5 s·log ω + β6 log s·log ω + β7 ω + β8 ω·log s
#
# Goal: estimate reduced-form mappings  βk = gk(θ), where θ are model
# parameters from the Parameters(...) call, excluding:
#   - fc
#   - inventory-grid parameters (Smax, Ns)
#   - scale
# ---------------------------------------------------------------

function estimate_v3_from_value(V_by_ω_local, Sgrid_local, ω_grid_local, skip_local)
    s_local = Sgrid_local[skip_local:end]
    Ns_local = length(s_local)
    Nω_local = length(ω_grid_local)

    s_stack_local = repeat(s_local, Nω_local)
    ω_stack_local = repeat(ω_grid_local, inner=Ns_local)
    log_s_local = log.(s_stack_local)
    log_ω_local = log.(ω_stack_local)

    y_local = vec(V_by_ω_local[skip_local:end, :])
    X_v3_local = hcat(ones(length(y_local)), s_stack_local, s_stack_local .^ 2,
                      log_s_local, log_ω_local,
                      s_stack_local .* log_ω_local,
                      log_s_local .* log_ω_local,
                      ω_stack_local,
                      ω_stack_local .* log_s_local)
    terms_v3_local = ["const", "s", "s^2", "log s", "log ω", "s·log ω", "log s·log ω", "ω", "ω·log s"]
    return fit_ols_model("V3", X_v3_local, y_local, terms_v3_local)
end

println("\n" * "="^70)
println("FINAL STEP: PARAMETER -> V3 COEFFICIENT MAPPING")
println("="^70)

v3_base = estimate_v3_from_value(V_by_omega, Sgrid, ω_grid, skip)
print_coef_table(v3_base, "BASELINE V3 COEFFICIENT TABLE")

param_names = ["c", "μη", "ση2", "ρ_ω", "γ", "δ", "β", "ϵ", "μν", "σν2", "size"]
Kθ = length(param_names)
Kβ = length(v3_base.β)

# Number of perturbed economies for mapping exercise
Nmap = 28
Random.seed!(1234)

Θ = zeros(Nmap, Kθ)
B = zeros(Nmap, Kβ)
ok = falses(Nmap)

for r in 1:Nmap
    # Controlled random perturbations around baseline
    c_r    = params.c    * (0.70 + 0.60 * rand())
    μη_r   = params.μη   + 0.25 * (2rand() - 1)
    ση2_r  = max(1e-4, params.ση2 * (0.60 + 0.90 * rand()))
    ρ_ω_r  = clamp(params.ρ_ω + 0.25 * (2rand() - 1), -0.7, 0.95)
    γ_r    = clamp(params.γ * (0.75 + 0.50 * rand()), 0.2, 1.4)
    δ_r    = clamp(params.δ * (0.60 + 0.90 * rand()), 0.005, 0.40)
    β_r    = clamp(params.β * (0.95 + 0.08 * rand()), 0.85, 0.995)
    ϵ_r    = clamp(params.ϵ * (0.70 + 0.70 * rand()), 1.2, 20.0)
    μν_r   = max(1e-3, params.μν * (0.70 + 0.70 * rand()))
    σν2_r  = max(1e-5, params.σν2 * (0.60 + 1.20 * rand()))
    size_r = max(1e-3, params.size * (0.70 + 0.70 * rand()))

    Θ[r, :] .= [c_r, μη_r, ση2_r, ρ_ω_r, γ_r, δ_r, β_r, ϵ_r, μν_r, σν2_r, size_r]

    try
        p_r = Parameters(c=c_r, fc=params.fc, μη=μη_r, ση2=ση2_r, ρ_ω=ρ_ω_r,
                         γ=γ_r, δ=δ_r, β=β_r, ϵ=ϵ_r, μν=μν_r, σν2=σν2_r,
                         Q=params.Q, Q_ω=params.Q_ω,
                         Smax=params.Smax, Ns=params.Ns,
                         scale=1.0, size=size_r)
        _, _, _, V_by_ω_r, _, _, _ = solve_model(p_r)
        v3_r = estimate_v3_from_value(V_by_ω_r, p_r.Sgrid, p_r.ω_grid, skip)
        B[r, :] .= v3_r.β
        ok[r] = true
    catch err
        @printf("  warning: sample %d failed (%s)\n", r, sprint(showerror, err))
    end
end

Θ_ok = Θ[ok, :]
B_ok = B[ok, :]
Nok = size(Θ_ok, 1)

println(@sprintf("\nSuccessful perturbed solves: %d / %d", Nok, Nmap))
if Nok < Kθ + 3
    println("Not enough successful samples for a stable mapping regression.")
else
    # Mapping basis: affine in levels + logs of positive-valued parameters.
    pos_cols = [1, 3, 5, 6, 7, 8, 9, 10, 11]  # c, ση2, γ, δ, β, ϵ, μν, σν2, size
    Z = hcat(ones(Nok), Θ_ok, log.(Θ_ok[:, pos_cols]))
    z_names = vcat(["const"],
                   ["lvl_" * n for n in param_names],
                   ["log_" * param_names[i] for i in pos_cols])

    println("\n" * "-"^70)
    println("MAPPING REGRESSIONS:  β_k(V3) = a_k + b_k' θ + d_k' log(θ_pos)")
    println("-"^70)

    for k in 1:Kβ
        yk = B_ok[:, k]
        θk = Z \ yk
        yhat_k = Z * θk
        sse_k = sum(abs2, yk - yhat_k)
        sst_k = sum(abs2, yk .- mean(yk))
        r2_k = 1.0 - sse_k / sst_k

        println(@sprintf("\nCoefficient mapping for V3 term %d (%s), R² = %.4f",
                         k, v3_base.terms[k], r2_k))
        println(@sprintf("%-18s %14s", "Mapping term", "Estimate"))
        println("-"^36)
        for j in eachindex(θk)
            @printf("%-18s %14.6f\n", z_names[j], θk[j])
        end
    end

    # Quick visual diagnostic: actual vs fitted mapping for each V3 coefficient
    plt_map = plot(layout=(3, 3), size=(1200, 900),
                   plot_title="Parameter-to-Coefficient Mapping Diagnostics (V3)")
    for k in 1:Kβ
        yk = B_ok[:, k]
        θk = Z \ yk
        yhat_k = Z * θk
        scatter!(plt_map[k], yk, yhat_k,
                 xlabel="Actual β", ylabel="Mapped β",
                 title=v3_base.terms[k], legend=false, markersize=3)
        lo = min(minimum(yk), minimum(yhat_k))
        hi = max(maximum(yk), maximum(yhat_k))
        plot!(plt_map[k], [lo, hi], [lo, hi], color=:black, linestyle=:dash, linewidth=1)
    end
    display(plt_map)
    savefig(plt_map, joinpath(@__DIR__, "..", "Notes", "AnalyticalVF_V3_parameter_mapping.png"))
    println("Saved mapping diagnostics figure → Notes/AnalyticalVF_V3_parameter_mapping.png")
end
