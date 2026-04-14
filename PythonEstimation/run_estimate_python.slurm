#!/bin/bash
#SBATCH --job-name=estimate-python
#SBATCH --output=logs/estimate-python-%j.out
#SBATCH --error=logs/estimate-python-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

# -----------------------------------------------------------------------------
# User-configurable paths and filenames
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-$HOME/work/Research/CRW_Inventory_Competition}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/SimulatedData}

TARGET_MOMENTS_FILE=${TARGET_MOMENTS_FILE:-target_moments_001.csv}
TARGET_VCOV_FILE=${TARGET_VCOV_FILE:-target_moment_vcov_001.csv}
GRID_FILE=${GRID_FILE:-moments.csv}
RESULTS_FILE=${RESULTS_FILE:-estimated_parameters_001.csv}

cd "$PROJECT_DIR"
mkdir -p logs


echo "SLURM job id: ${SLURM_JOB_ID:-unknown}"
echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "Input files: $TARGET_MOMENTS_FILE, $TARGET_VCOV_FILE, $GRID_FILE"
echo "Output file: $RESULTS_FILE"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown} MB per node"

module load python

python -m PythonEstimation.estimate_run \
        --data-dir "$DATA_DIR" \
        --target-moments-file "$TARGET_MOMENTS_FILE" \
        --target-vcov-file "$TARGET_VCOV_FILE" \
        --grid-file "$GRID_FILE" \
        --results-file "$RESULTS_FILE" 