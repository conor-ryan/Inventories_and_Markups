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
#SBATCH --array=1-4
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
cd ~/work/Research/CRW_Inventory_Competition/

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

SIM_DATA_DIR="../../SimulatedData"
SUMMARY_OUT="${SIM_DATA_DIR}/monte_carlo_summary.csv"
SUMMARY_DONE="${SIM_DATA_DIR}/.monte_carlo_summary.done"
SUMMARY_LOCK="${SIM_DATA_DIR}/.monte_carlo_summary.lock"

python estimation_main.py \
    --id "${DATASET_ID}" \
    --n-threads "${SLURM_CPUS_PER_TASK:-1}" \
    --sim-data-dir "${SIM_DATA_DIR}"

# After each task finishes, one task will create the aggregate summary once
# all expected estimates_id_*.csv files are present.
if [[ -n "${MC_EXPECTED_TASKS:-}" ]]; then
    EXPECTED_TASKS="${MC_EXPECTED_TASKS}"
elif [[ -n "${SLURM_ARRAY_TASK_MIN:-}" && -n "${SLURM_ARRAY_TASK_MAX:-}" ]]; then
    ARRAY_STEP="${SLURM_ARRAY_TASK_STEP:-1}"
    EXPECTED_TASKS=$(( (SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN) / ARRAY_STEP + 1 ))
else
    EXPECTED_TASKS=1
fi

mkdir -p "${SIM_DATA_DIR}"
exec 9>"${SUMMARY_LOCK}"
if flock -n 9; then
    if [[ -f "${SUMMARY_DONE}" ]]; then
        echo "Summary already created: ${SUMMARY_OUT}"
    else
        N_EST=$(find "${SIM_DATA_DIR}" -maxdepth 1 -name 'estimates_id_*.csv' | wc -l)
        if [[ "${N_EST}" -ge "${EXPECTED_TASKS}" ]]; then
            echo "All estimates found (${N_EST}/${EXPECTED_TASKS}); building summary..."
            python summarize_monte_carlo.py \
                --sim-data-dir "${SIM_DATA_DIR}" \
                --output-path "${SUMMARY_OUT}"
            touch "${SUMMARY_DONE}"
            echo "Summary complete: ${SUMMARY_OUT}"
        else
            echo "Summary deferred (${N_EST}/${EXPECTED_TASKS} estimates present)."
        fi
    fi
fi
