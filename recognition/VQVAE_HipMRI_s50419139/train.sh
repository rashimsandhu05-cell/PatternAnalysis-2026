#!/bin/bash
# Slurm batch script for training on Rangpur.
#
# Usage (from this folder, on the Rangpur login node):
#   sbatch train.sh vqvae                  # main model
#   sbatch train.sh vae                    # baseline
#   sbatch train.sh pixelcnn               # prior (after the vqvae job has finished)
#   sbatch train.sh vqvae --epochs 60      # extra arguments are passed to train.py
#   sbatch -p a100 train.sh vae            # use a different GPU partition
#
# Monitor:  squeue -u $USER        Output:  tail -f slurm-<jobid>.out
# Cancel:   scancel <jobid>

#SBATCH --job-name=hipmri-gen
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# (no "set -u": conda's activation scripts reference unset variables)
set -eo pipefail

MODEL=${1:?"usage: sbatch train.sh <vqvae|vae|pixelcnn> [train.py args]"}
shift

# Activate the conda environment in a non-interactive shell.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate comp3710

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

python train.py --model "$MODEL" "$@"

echo "Finished at $(date)"
