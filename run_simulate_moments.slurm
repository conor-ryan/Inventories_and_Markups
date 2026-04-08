#!/bin/bash
#SBATCH --job-name=simulate-moments
#SBATCH --output=logs/simulate-moments-%j.out
#SBATCH --error=logs/simulate-moments-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

# Run from the directory containing this script.
cd ~/work/Research/CRW_Inventory_Competition/


# If your cluster uses environment modules, uncomment and adjust:
module load julia

JULIA_BIN=${JULIA_BIN:-julia}
THREADS=${SLURM_CPUS_PER_TASK:-1}
${JULIA_BIN} --threads=${THREADS} Code/SimulateMoments.jl
