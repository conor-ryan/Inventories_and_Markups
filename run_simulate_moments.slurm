#!/bin/bash
#SBATCH --job-name=simulate-moments
#SBATCH --output=logs/simulate-moments-%j.out
#SBATCH --error=logs/simulate-moments-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Run from the directory containing this script.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/logs"

# If your cluster uses environment modules, uncomment and adjust:
module load julia

JULIA_BIN=${JULIA_BIN:-julia}
THREADS=${SLURM_CPUS_PER_TASK:-1}

if [[ -f "${SCRIPT_DIR}/Project.toml" ]]; then
  ${JULIA_BIN} --project="${SCRIPT_DIR}" --threads=${THREADS} -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
  ${JULIA_BIN} --project="${SCRIPT_DIR}" --threads=${THREADS} SimulateMoments.jl
else
  echo "Project.toml not found in Code. Running without --project." >&2
  ${JULIA_BIN} --threads=${THREADS} SimulateMoments.jl
fi