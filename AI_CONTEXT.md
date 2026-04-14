# AI Editing Context

Purpose: shared notes for AI assistants editing this project.

## Project Scope
- Dynamic inventory model and indirect inference estimation in Julia.
- Core workflow: solve model -> simulate panel data -> compute moments -> estimate parameters -> compute standard errors.

## Core Files
- Model primitives and simulation internals: ModelFunctions.jl
- Estimation and moments logic: EstimationFunctions.jl
- Panel simulation and bootstrap utilities: SimulationFunctions.jl
- End-to-end estimator from saved moment files: EstimateFromSavedMoments.jl
- Legacy all-in-one runner (may be out of date): RunEstimation.jl


## Editing Notes
- Prefer minimal, local edits.
- Keep parameter ordering and moment ordering unchanged unless intentionally refactoring all dependent code.
- When changing estimator signatures, update all script call sites and docstrings in the same pass.
- Never introduce hat/caret diacritics in identifiers or field names (for example: γ̂, ϵ̂, δ̂, m̂, m̃). These hats are unnecessary and can be simply labelled with the greek character.
- Do not over-use defensive error checking or robustness guards. In this project, failures can be useful diagnostics for misspecification and should not be suppressed by default.

## Session Log
- 2026-04-14: Added this shared AI context file.

## Python Translation Task (PythonEstimation)
- Objective: create a Python implementation in `PythonEstimation/` that executes the same end-to-end logic as `Code/Estimate_Run.jl`.
- Scope for translation: core logic from `Code/Estimate_Run.jl`, `Code/ModelFunctions.jl`, and `Code/EstimationFunctions.jl` that is required to:
	1. load target moments / vcov / precomputed grid,
	2. select the best grid start,
	3. estimate 7 structural parameters by indirect inference,
	4. compute asymptotic variance / standard errors,
	5. save estimated parameters to CSV.

### Required Python Module Mapping
- `PythonEstimation/model_functions.py`:
	- `Parameters` container and constructors
	- Tauchen discretization / stationary distribution for omega process
	- value-function solver and policy interpolation helpers
	- simulation routines used by estimation moments
- `PythonEstimation/estimation_functions.py`:
	- annual auxiliary regressions and AR(1) helper
	- monthly and annual moment computation
	- grid-start selector
	- full-II objective and optimizer wrapper
	- numerical Jacobian and asymptotic variance routine
- `PythonEstimation/estimate_run.py`:
	- Python equivalent of `Code/Estimate_Run.jl` script entry point.

### Numerical/Behavioral Equivalence Requirements
- Keep parameter ordering unchanged: `(γ, μη, ση2, ρω, σν2, ϵ, δ)`.
- Keep moment ordering unchanged: `(avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, μ_η)`.
- Keep objective definition unchanged: `M' W M`, with `M = m_hat - m_tilde`.
- Preserve transformation/bounds logic used in optimization (log, tanh, logit parameterizations).
- Preserve random-seed behavior where feasible so repeated runs are reproducible.

### Performance Priorities For Python Port
- Prioritize vectorized NumPy operations in hot loops where possible.
- Use SciPy interpolation and optimization primitives analogous to Julia usage.
- Cache/reuse precomputed arrays (quadrature nodes, demand tables, transition objects) in value-function and simulation routines.
- Maintain functional parity first, then optimize hotspots without changing outputs.
- Treat current Julia speed choices as the starting baseline, not as fixed requirements; for each hotspot, evaluate whether Python-specific alternatives are faster/more stable (for example NumPy vectorization, Numba/JAX acceleration, SciPy solver choices, or different interpolation/caching layouts).


### ModelFunctions Speed Notes (Existing Julia Optimizations)
- `UniformInterp` is used for uniform-grid linear interpolation with direct index arithmetic (`t = (x - s_lo) * inv_h`) to avoid binary-search overhead from generic interpolators in hot loops.
- `OmegaPolicyInterp` stores per-omega interpolation objects so policy evaluation in simulation/value iteration is a cheap call.
- `precompute_demand` builds `D_table` and `C_table` over `(q, j, i)` to remove repeated power calls (`p^(-ϵ)`, `D^γ`) inside inner quadrature evaluations.
- In `solve_value_function`:
	- `mul!(EV_cont_j, V_by_omega, P_ω[j, :])` is used to compute continuation values in-place.
	- `EV_cont_j` is preallocated once and reused across iterations.
	- a monotone-like warm bound (`n_upper`) is updated across inventory grid points to tighten 1D Brent search bounds.
- Fast path (`full=false`) uses 1D optimizations with fixed-price policy and precomputed demand tables before optional full joint `(n,p)` optimization.
- `build_fast_policy_interpolants` prebuilds interpolation nodes once after solving, reducing runtime overhead in simulation and moment generation.
- Simulation uses integer omega-state transitions (`draw_ω_index`) with row-wise cumulative probability draws from `P_ω` instead of continuous shock interpolation each period.

### Deliverables
- Python package-style code under `PythonEstimation/` implementing the translated logic.
- A README in `PythonEstimation/README.md` documenting:
	- file/module structure,
	- required Python dependencies,
	- command to run the estimator,
	- expected input/output files,
	- notes on parity vs Julia implementation.

- 2026-04-14: Added Python translation specifications for the `PythonEstimation` implementation of the full `Estimate_Run.jl` workflow.
- 2026-04-14: Added notes documenting existing speed optimizations in `ModelFunctions.jl` to preserve these design choices in Python translation.
- 2026-04-14: Clarified that Julia performance design is a baseline; Python implementation should re-evaluate and profile potentially better Python-native optimization strategies.
