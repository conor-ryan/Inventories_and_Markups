#!/bin/bash
#SBATCH --job-name=simulate-moments
#SBATCH --output=logs/simulate-moments-%j.out
#SBATCH --error=logs/simulate-moments-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

set -euo pipefail

# Run from the directory containing this script.
cd ~/work/Research/CRW_Inventory_Competition/


# If your cluster uses environment modules, uncomment and adjust:
module load julia

JULIA_BIN=${JULIA_BIN:-julia}
THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "SLURM job id: ${SLURM_JOB_ID:-unknown}"
echo "Requested CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Requested memory: ${SLURM_MEM_PER_NODE:-unknown} MB per node"

echo "Memory snapshot before run:"
free -h || true

if [[ -x /usr/bin/time ]]; then
	/usr/bin/time -v ${JULIA_BIN} --threads=${THREADS} Code/SimulateMoments.jl
else
	echo "/usr/bin/time not found; running without process-level peak RSS report."
	${JULIA_BIN} --threads=${THREADS} Code/SimulateMoments.jl
fi

echo "Memory snapshot after run:"
free -h || true
