#!/bin/bash
#SBATCH --job-name=simulate-and-save-moments
#SBATCH --output=logs/simulate-and-save-moments-%j.out
#SBATCH --error=logs/simulate-and-save-moments-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G

set -euo pipefail

# Run from the repository root expected by this workflow.
cd ~/work/Research/CRW_Inventory_Competition/

# If your cluster uses environment modules, adjust as needed.
module load julia

JULIA_BIN=${JULIA_BIN:-julia}
THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "SLURM job id: ${SLURM_JOB_ID:-unknown}"
echo "Requested CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Requested memory: ${SLURM_MEM_PER_NODE:-unknown} MB per node"

echo "Memory snapshot before run:"
free -h || true

if [[ -x /usr/bin/time ]]; then
	/usr/bin/time -v ${JULIA_BIN} --threads=${THREADS} Code/SimulateAndSaveMoments.jl
else
	echo "/usr/bin/time not found; running without process-level peak RSS report."
	${JULIA_BIN} --threads=${THREADS} Code/SimulateAndSaveMoments.jl
fi

echo "Memory snapshot after run:"
free -h || true
