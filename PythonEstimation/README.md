### Description of /PythonEstimation code object
The main code for the model is written in julia, contained in Code/
- ModelFunctions.jl contains the main methods for the model
- EstimationFunctions.jl contains functions used to estimate the model
- EstimateRun.jl loads this code and estimates the model from data. 

The overall objective of the code in this subfolder is to replicate this code in python. 

### Replication Stage 1 ###
The first task is to replicate ModelFunctions.jl. The most important method is solve_value_function, which uses value function iteration to solve the model given a set of paramaters. 

The first obective is to replciate this function and check it that it produces the same results given the same parameters. Make sure the parameter defaults are the same as specified in ModelFunctions.jl. Test the model at the parameters defined on line 6 of SolveModel.jl. 

### Replication Stage 2 ###
The second task is to replicate EstimationFunctions.jl. The ultimate goal is to be able to estimate the model, as is Code/Estimate_Run.jl. 

In replicating this function, it is important to pay special attention to simulating the model using the same logic (_simulate_all_moments) but with modifcations so that it is optimized for computational speed in python. Estimate_params_ii_full is another important function which uses NelderMead to minimize the objective function. If possible, the python version should use similar (NLOpt) optimizers. 

Write all new functions into estimation_functions.py. One important function, (simulate_firm()), is in Code/ModelFunctions.jl, but this can be included in estimation_functions.py. 

#### Stage 2 Implementation Plan ####

All code goes in `estimation_functions.py`. Add corresponding interactive blocks to `interactive.py` after each step is complete.

**Step 1 — `simulate_firm` (Julia: `ModelFunctions.jl::simulate_firm`)**
- Inputs: `n_firms, n_months, p_policy (ns, n_omega), n_policy (ns, n_omega), params, seed, burn_in=100`
- Logic: for each firm draw initial `s` from sgrid (uniform), initial `omega_idx` from `pi_omega` (ergodic); run burn-in then record months
- At each step: look up `p = p_policy[s_idx, omega_idx]`, `n = n_policy[s_idx, omega_idx]` via uniform-grid interpolation (same as Stage 1); draw `nu ~ LogNormal`; compute `D = min(nu * p^{-eps}, s)`; advance `s = (1-delta)*(s - D + n)`; transition `omega_idx` by inverse-CDF on `p_omega` row
- Speed: pre-draw ALL random numbers (`nu` draws and `omega` uniform draws) outside Numba as NumPy arrays; pass as arguments; compile the per-firm loop with `@jit(nopython=True, parallel=True)` using `prange` over firms
- Returns: flat arrays `inv_sim (n_firms*n_months,)`, `dem_sim`, `rev_sim`, `exp_sim` (operating expenses = omega * D^gamma)

**Step 2 — moment computation (Julia: `EstimationFunctions.jl::compute_monthly_moments` + `compute_annual_auxiliary`)**
- Monthly moments (pure NumPy, vectorized):
  - `avg_isr`: mean of `inv_sim / rev_sim` (skip zeros)
  - `var_log1p_isr`: variance of `log1p(inv_sim / rev_sim)`
  - `avg_gross_margin`: mean of `rev_sim / (c * dem_sim)`
- Annual aggregation: reshape to `(n_firms, n_years, 12)` and sum along axis=2 — no DataFrame needed
- Annual auxiliary stats via NumPy OLS (2-column design matrix + `np.linalg.lstsq`) and AR(1) (`estimate_omega_ar1`)
- Returns named tuple / dict with 7 moments: `avg_isr, var_log1p_isr, avg_gross_margin, gamma_OLS, rho_omega, sigma_eta2, mu_eta`

**Step 3 — `_simulate_all_moments` (Julia: `EstimationFunctions.jl::_simulate_all_moments`)**
- Thin wrapper: calls `simulate_firm`, then Step 2 moment functions
- Inputs: `params, p_policy, n_policy, n_firms, n_years, seed`
- Returns dict of 7 simulated moments

**Step 4 — parameter packing / unpacking**
- `_pack(theta)`: maps `[gamma, mu_eta, sigma_eta2, rho_omega, sigma_nu2, epsilon, delta]` to unconstrained search vector `[gamma, mu_eta, log(sigma_eta2), arctanh(rho_omega), log(sigma_nu2), epsilon, logit(delta)]`
- `_unpack(x)`: inverse with clamping to bounds (matches Julia's `unpack` in `estimate_params_ii_full`)

**Step 5 — `estimate_params_ii_full` (Julia: `EstimationFunctions.jl::estimate_params_ii_full`)**
- Inputs: `target_moments` (dict), `init_guess` (length-7 list), `W` (7x7 array), keyword args for `n_firms, n_years, seed, max_iter, verbose`
- Objective: solve model at current params → simulate → compute `(m_hat - m_tilde) @ W @ (m_hat - m_tilde)`
- Optimizer: `scipy.optimize.minimize(..., method='Nelder-Mead')` with `options={'maxiter': max_iter}`
- Returns dict with estimated params, objective value, and optimizer result

**Step 6 — `select_best_grid_start` (Julia: `EstimationFunctions.jl::select_best_grid_start`)**
- Inputs: `df_grid` (pandas DataFrame or path to CSV), `target_moments` dict, `W`
- Pure NumPy: compute weighted objective for all non-failed rows at once, return argmin row
- Returns dict with best parameter values and objective value

**Step 7 — `interactive.py` Stage 2 blocks**
- Block: load target moments and weighting matrix from CSV
- Block: run `select_best_grid_start` to get warm start
- Block: run `estimate_params_ii_full` with timing
- Block: print estimated parameters

#### Working Preference ####
User prefers line-by-line interactive Python terminal execution over command-line scripts.
Use `interactive.py` for this workflow: copy-paste sections into a Python terminal.


#### Coding Guidelines ####
1) Do not use robustness error checking methods that might slow down performance. 
2) Assume that the user will always provide functions with correct inputs
3) Speed is important. Code should use native features of python that encourage speed rather than the original logic of julia. 
4) The full model solution method is not necessary. 
5) Interpolations/extrapolations are an important part of speed. These methods need to be as efficient as possible. 

#### Speed Guidelines ####
1) Where possible, make sure to vectorize the code using Numpy. It will not be vectorized in julia. You must check each function to ask whether vectorization is possible. 

### Workspace Added For Replication ###
Replication files are now in the PythonEstimation root:
- `model_functions.py`: Python translation of the core model pieces needed for value-function iteration.
- `interactive.py`: block-by-block workflow for Python terminal execution using the target parameters from line 6 of `SolveModel.jl`.
- `VFI_REPLICATION.md`: short guide for scope and usage.

### Parity Checking ###
Use these files to compare Julia and Python outputs for the same objects:
- `export_julia_reference.jl`: runs the Julia model and saves reference arrays to `parity_reference/`.
- `check_parity.py`: runs the Python replication, loads Julia references, and prints max absolute and relative differences.

Suggested run order:
1) In Julia: `include("PythonEstimation/export_julia_reference.jl")`
2) In Python: `python check_parity.py`

Or run from Python only:
- `python check_parity.py --run-julia`
