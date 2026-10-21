#!/bin/bash
# Slurm batch script for evaluation on Rangpur.
#   sbatch predict.sh                       # all tasks
#   sbatch predict.sh --tasks evaluate      # extra arguments are passed to predict.py
#SBATCH --job-name=hipmri-eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-eval-%j.out

# (no "set -u": conda's activation scripts reference unset variables)
set -eo pipefail
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate comp3710
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
python predict.py "$@"
echo "Finished at $(date)"
