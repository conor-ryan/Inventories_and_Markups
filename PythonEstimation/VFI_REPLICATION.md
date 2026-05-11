# VFI Replication Workspace

This folder is the Python work area for replicating the Julia value-function solver in ModelFunctions.jl.

## Files

- model_functions.py: Python translation of the key model pieces needed to run value-function iteration.
- interactive.py: line-by-line terminal workflow with the target parameters from SolveModel.jl line 6.

## Scope

- Replicates the solve_value_function(..., full=false, fast_interp=true) branch.
- Keeps parameter defaults aligned with the Julia Parameters constructor.
- Uses fast uniform-grid interpolation for continuation values.

## Usage

Open a Python terminal in this folder and run interactive.py block by block.
