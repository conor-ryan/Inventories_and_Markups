### Description of /PythonEstimation code object
The main code for the model is written in julia, contained in Code/
- ModelFunctions.jl contains the main methods for the model
- EstimationFunctions.jl contains functions used to estimate the model
- EstimateRun.jl loads this code and estimates the model from data. 

The overall objective of the code in this subfolder is to replicate this code in python. 

### Replication Stage 1 ###
The first task is to replicate ModelFunctions.jl. The most important method is solve_value_function, which uses value function iteration to solve the model given a set of paramaters. 

The first obective is to replciate this function and check it that it produces the same results given the same parameters. Make sure the parameter defaults are the same as specified in ModelFunctions.jl. Test the model at the parameters defined on line 6 of SolveModel.jl. 

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
