#!/bin/bash
#SBATCH --job-name=fire-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load conda
source activate fire-risk

# Navigate to the project root directory
cd "${SLURM_SUBMIT_DIR:-.}"

set -euo pipefail

# --- offline wandb (sync later from a login node with internet) ---
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-fire-risk}"
export WANDB_DIR="${WANDB_DIR:-./runs}"

mkdir -p logs runs

# --- run ---
RUN_NAME="${RUN_NAME:-baseline-$(date +%Y%m%d-%H%M%S)}"
echo "Running ${RUN_NAME}"
nvidia-smi || true

python -m src.train_baseline \
    --data-dir data/baseline/clean \
    --run-name "${RUN_NAME}" \
    --out-dir runs \
    --device cuda

echo "Done. To sync wandb runs from a node with internet:"
echo "  wandb sync ./runs/wandb/offline-run-*"
