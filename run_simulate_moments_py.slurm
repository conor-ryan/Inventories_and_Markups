#!/bin/bash
#SBATCH --job-name=simulate-moments
#SBATCH --output=logs/simulate-moments-%A_%a.out
#SBATCH --error=logs/simulate-moments-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --account=econ_crch_research
#SBATCH --partition=basic
#SBATCH --array=1-4
set -euo pipefail

# Run from the directory containing this script.
cd ~/work/Research/CRW_Inventory_Competition

# Load Python environment (matches run_monte_carlo.slurm pattern).
module load anaconda
conda activate inventory

echo "SLURM job id:    ${SLURM_JOB_ID:-unknown}"
echo "Array task id:   ${SLURM_ARRAY_TASK_ID:-unknown} / ${SLURM_ARRAY_TASK_COUNT:-unknown}"
echo "Requested CPUs:  ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Requested memory: ${SLURM_MEM_PER_NODE:-unknown} MB per node"

echo "Memory snapshot before run:"
free -h || true

if [[ -x /usr/bin/time ]]; then
	/usr/bin/time -v python Code/PythonEstimation/simulatemoments.py
else
	echo "/usr/bin/time not found; running without process-level peak RSS report."
	python Code/PythonEstimation/simulatemoments.py
fi

echo "Memory snapshot after run:"
free -h || true

# After the last array task completes, submit a small concatenation job.
# This only runs once (when SLURM_ARRAY_TASK_ID equals the last task).
if [[ "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_COUNT}" ]]; then
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
           --job-name=concat-moments \
           --output=logs/concat-moments-%j.out \
           --error=logs/concat-moments-%j.err \
           --time=00:10:00 \
           --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
           --account=econ_crch_research \
           --partition=basic \
           --wrap="source activate inventory && python Code/PythonEstimation/concat_moments.py"
    echo "Concatenation job submitted (depends on array job ${SLURM_ARRAY_JOB_ID})."
fi
