using Distributions, LinearAlgebra, Optim, FastGaussQuadrature, Plots, Interpolations, LineSearch, Random, Statistics, DataFrames, Printf
include("ModelFunctions.jl")

params = Parameters(c=1.0, fc=0.0, μη=log(0.1), ση2=0.05, ρ_ω=0.1, γ=0.9, δ=0.05, β=0.95, ϵ=8.0, μν=1, σν2=0.09, Smax=20, Ns=200, scale=1.0, size=100)

Sgrid = params.Sgrid

# ---------------------------------------------------
# Solve with initial algorithm (full=false)
# ---------------------------------------------------
println("Solving with initial algorithm (full=false)...")
p_init, order_init, V_init, V_by_omega_init, _, _, _ = solve_model(params, full=false)

# ---------------------------------------------------
# Solve with full algorithm (full=true)
# ---------------------------------------------------
println("Solving with full algorithm (full=true)...")
p_full, order_full, V_full, V_by_omega_full, _, _, _ = solve_model(params, full=true)

# ---------------------------------------------------
# Omega-integrated policies (using ergodic distribution)
# ---------------------------------------------------
ω_weights = params.π_ω

p_init_int    = p_init    * ω_weights
p_full_int    = p_full    * ω_weights
order_init_int = order_init * ω_weights
order_full_int = order_full * ω_weights

# ---------------------------------------------------
# Difference diagnostics
# ---------------------------------------------------
@printf("\n=== Max absolute differences (initial vs full) ===\n")
@printf("Price policy:     %.6f\n", maximum(abs.(p_full_int    .- p_init_int)))
@printf("Order policy:     %.6f\n", maximum(abs.(order_full_int .- order_init_int)))
@printf("Value function:   %.6f\n", maximum(abs.(V_full         .- V_init)))

# ---------------------------------------------------
# Omega slices for per-omega comparison
# ---------------------------------------------------
Q = params.Q_ω
j_low  = 1
j_med  = (Q + 1) ÷ 2
j_high = Q

# ---------------------------------------------------
# Comparison plots  (3 rows × 3 columns)
#   Rows:    price policy | order policy | value function
#   Col 1:   E_ω comparison (initial vs full)
#   Col 2:   ω slices (low / med / high) for initial and full
#   Col 3:   difference (full − initial) for each ω slice
# ---------------------------------------------------

slice_colors   = [:blue :green :red]
slice_labels_i = ["Init ω_low" "Init ω_med" "Init ω_high"]
slice_labels_f = ["Full ω_low" "Full ω_med" "Full ω_high"]
diff_labels    = ["Δ ω_low" "Δ ω_med" "Δ ω_high"]

# ── Row 1: Price policy ──────────────────────────────────────────────────────

# Col 1 – E_ω
pc11 = plot(Sgrid[2:end], [p_init_int[2:end] p_full_int[2:end]],
    xlabel="Inventory (s)", ylabel="Price",
    title="Price: E_ω",
    label=["Initial" "Full"],
    linewidth=2, linestyle=[:solid :dash], color=[:black :black])

# Col 2 – ω slices
pc12 = plot(Sgrid[2:end],
    hcat(p_init[2:end, j_low], p_init[2:end, j_med], p_init[2:end, j_high]),
    xlabel="Inventory (s)", ylabel="Price",
    title="Price: ω slices",
    label=slice_labels_i, linewidth=2, linestyle=:solid, color=slice_colors)
plot!(pc12, Sgrid[2:end],
    hcat(p_full[2:end, j_low], p_full[2:end, j_med], p_full[2:end, j_high]),
    label=slice_labels_f, linewidth=2, linestyle=:dash, color=slice_colors)

# Col 3 – differences
pc13 = plot(Sgrid[2:end],
    hcat(p_full[2:end, j_low] .- p_init[2:end, j_low],
         p_full[2:end, j_med] .- p_init[2:end, j_med],
         p_full[2:end, j_high] .- p_init[2:end, j_high]),
    xlabel="Inventory (s)", ylabel="Full − Initial",
    title="Price: difference",
    label=diff_labels, linewidth=2, color=slice_colors)
hline!(pc13, [0.0], linewidth=1, linestyle=:dot, color=:black, label="")

# ── Row 2: Order policy ──────────────────────────────────────────────────────

# Col 1 – E_ω
pc21 = plot(Sgrid, [order_init_int order_full_int],
    xlabel="Inventory (s)", ylabel="Orders",
    title="Orders: E_ω",
    label=["Initial" "Full"],
    linewidth=2, linestyle=[:solid :dash], color=[:black :black])

# Col 2 – ω slices
pc22 = plot(Sgrid,
    hcat(order_init[:, j_low], order_init[:, j_med], order_init[:, j_high]),
    xlabel="Inventory (s)", ylabel="Orders",
    title="Orders: ω slices",
    label=slice_labels_i, linewidth=2, linestyle=:solid, color=slice_colors)
plot!(pc22, Sgrid,
    hcat(order_full[:, j_low], order_full[:, j_med], order_full[:, j_high]),
    label=slice_labels_f, linewidth=2, linestyle=:dash, color=slice_colors)

# Col 3 – differences
pc23 = plot(Sgrid,
    hcat(order_full[:, j_low] .- order_init[:, j_low],
         order_full[:, j_med] .- order_init[:, j_med],
         order_full[:, j_high] .- order_init[:, j_high]),
    xlabel="Inventory (s)", ylabel="Full − Initial",
    title="Orders: difference",
    label=diff_labels, linewidth=2, color=slice_colors)
hline!(pc23, [0.0], linewidth=1, linestyle=:dot, color=:black, label="")

# ── Row 3: Value function ────────────────────────────────────────────────────

# Col 1 – E_ω
pc31 = plot(Sgrid, [V_init V_full],
    xlabel="Inventory (s)", ylabel="Value",
    title="Value: E_ω",
    label=["Initial" "Full"],
    linewidth=2, linestyle=[:solid :dash], color=[:black :black])

# Col 2 – ω slices
pc32 = plot(Sgrid,
    hcat(V_by_omega_init[:, j_low], V_by_omega_init[:, j_med], V_by_omega_init[:, j_high]),
    xlabel="Inventory (s)", ylabel="Value",
    title="Value: ω slices",
    label=slice_labels_i, linewidth=2, linestyle=:solid, color=slice_colors)
plot!(pc32, Sgrid,
    hcat(V_by_omega_full[:, j_low], V_by_omega_full[:, j_med], V_by_omega_full[:, j_high]),
    label=slice_labels_f, linewidth=2, linestyle=:dash, color=slice_colors)

# Col 3 – differences
pc33 = plot(Sgrid,
    hcat(V_by_omega_full[:, j_low] .- V_by_omega_init[:, j_low],
         V_by_omega_full[:, j_med] .- V_by_omega_init[:, j_med],
         V_by_omega_full[:, j_high] .- V_by_omega_init[:, j_high]),
    xlabel="Inventory (s)", ylabel="Full − Initial",
    title="Value: difference",
    label=diff_labels, linewidth=2, color=slice_colors)
hline!(pc33, [0.0], linewidth=1, linestyle=:dot, color=:black, label="")

# ── Combined ─────────────────────────────────────────────────────────────────

combined_plot = plot(pc11, pc12, pc13,
                     pc21, pc22, pc23,
                     pc31, pc32, pc33,
                     layout=(3, 3), size=(1800, 1400), margin=5Plots.mm)
display(combined_plot)
