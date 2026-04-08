#!/bin/bash
#SBATCH --job-name=simulate-moments-array
#SBATCH --output=logs/sim-mom-array-%A_%a.out
#SBATCH --error=logs/sim-mom-array-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-20

set -euo pipefail

# Hard-coded project directory on cluster.
cd ~/work/Research/CRW_Inventory_Competition/Code/
mkdir -p logs
module load julia

JULIA_BIN=${JULIA_BIN:-julia}
THREADS=${SLURM_CPUS_PER_TASK:-1}
${JULIA_BIN} --threads=${THREADS} SimulateMomentsSlurm.jl

