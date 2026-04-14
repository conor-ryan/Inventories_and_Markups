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
