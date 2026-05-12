#!/bin/bash
# =============================================================================
# Monte Carlo estimation — SLURM array job
#
# SUBMITTING:
#   Default (10 datasets):
#       sbatch run_monte_carlo.slurm
#
#   Override dataset count (e.g. 25):
#       sbatch --array=1-25 run_monte_carlo.slurm
#
#   Auto-detect from SimulatedData directory:
#       N=$(ls ../../SimulatedData/target_moments_id_*.csv | wc -l)
#       sbatch --array=1-${N} run_monte_carlo.slurm
#
#   Throttle to at most K concurrent jobs (e.g. K=4):
#       sbatch --array=1-${N}%4 run_monte_carlo.slurm
# =============================================================================
#SBATCH --job-name=mc-estimation
#SBATCH --array=1-49
#SBATCH --output=logs/mc-%A_%a.out
#SBATCH --error=logs/mc-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --account=econ_crch_research
#SBATCH --partition=basic
set -euo pipefail

# Run from the Code directory (mirrors run_simulate_moments.slurm convention).
cd ~/work/Research/CRW_Inventory_Competition/Code
git pull
cd ..

module load anaconda
conda activate inventory

DATASET_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})

echo "========================================"
echo "SLURM job id     : ${SLURM_JOB_ID:-unknown}"
echo "Array task id    : ${SLURM_ARRAY_TASK_ID}"
echo "Dataset id       : ${DATASET_ID}"
echo "CPUs per task    : ${SLURM_CPUS_PER_TASK:-1}"
echo "========================================"

cd ~/work/Research/CRW_Inventory_Competition/Code/PythonEstimation

python estimation_main.py \
    --id "${DATASET_ID}" \
    --n-threads "${SLURM_CPUS_PER_TASK:-1}" \
    --sim-data-dir "../../SimulatedData"
