# PythonEstimation

Python translation of the Julia indirect-inference workflow used by `Code/Estimate_Run.jl`.

## What This Contains

- `model_functions.py`: translated model primitives, value-function solver, interpolation helpers, and simulation logic.
- `estimation_functions.py`: translated moment construction, full indirect-inference objective, grid start selection, Jacobian, and asymptotic variance.
- `estimate_run.py`: end-to-end runner that mirrors the Julia script `Code/Estimate_Run.jl` using saved moments and covariance files in `SimulatedData/`.
- `profile_solve_model.py`: bottleneck/profiling script focused on `solve_model` with a candidate parameter vector and optional per-iteration value-function logs.

## Dependencies

Install with pip:

```bash
pip install numpy pandas scipy
```

## Run

From `Code/`:

```bash
python -m PythonEstimation.estimate_run
```

Profile `solve_model` bottlenecks:

```bash
python -m PythonEstimation.profile_solve_model --verbose --ns 120 --maxiter 300 --profile-lines 30
```

Optional filename/path overrides:

```bash
python -m PythonEstimation.estimate_run \
  --data-dir ../SimulatedData \
  --target-moments-file target_moments.csv \
  --target-vcov-file target_moment_vcov.csv \
  --grid-file moments.csv \
  --results-file estimated_parameters.csv
```

## Slurm Run (Linux)

Use `run_estimate_python.slurm` in this folder.

Example:

```bash
sbatch Code/PythonEstimation/run_estimate_python.slurm
```

You can override filenames and directories at submit time:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=/path/to/CRW_inventory_competition,DATA_DIR=/path/to/CRW_inventory_competition/SimulatedData,TARGET_MOMENTS_FILE=target_moments.csv,TARGET_VCOV_FILE=target_moment_vcov.csv,GRID_FILE=moments.csv,RESULTS_FILE=estimated_parameters_py.csv \
  Code/PythonEstimation/run_estimate_python.slurm
```

## Inputs And Outputs

- Inputs read from `SimulatedData/`:
  - `target_moments.csv`
  - `target_moment_vcov.csv`
  - `moments.csv`
- Output written to `SimulatedData/`:
  - `estimated_parameters.csv`

## Notes On Parity

- Parameter and moment ordering follow the Julia implementation:
  - Parameters: `(γ, μη, ση2, ρω, σν2, ϵ, δ)`
  - Moments: `(avg_isr, var_log1p_isr, avg_gross_margin, γ_OLS, ρ_ω, σ_η2, μ_η)`
- The efficient path used by `Estimate_Run.jl` is implemented (`solve_model(..., full=False)`).
- `full=True` in the value-function solver is currently left unimplemented in Python because the run script does not use it.
- Julia speed ideas were used as a baseline (uniform-grid interpolation, demand precomputation), but final optimization choices should be validated with Python profiling.
