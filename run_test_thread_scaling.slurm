#!/bin/bash
#SBATCH --job-name=thread-scaling
#SBATCH --output=logs/thread-scaling-%j.out
#SBATCH --error=logs/thread-scaling-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --account=econ_crch_research
#SBATCH --partition=basic
set -euo pipefail

cd ~/work/Research/CRW_Inventory_Competition

module load anaconda
conda activate inventory

echo "SLURM job id:   ${SLURM_JOB_ID}"
echo "Requested CPUs: ${SLURM_CPUS_PER_TASK}"
echo ""

python Code/PythonEstimation/test_thread_scaling.py
