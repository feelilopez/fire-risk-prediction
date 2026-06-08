#!/bin/bash
#SBATCH --job-name=fire-lstm-focal-dual
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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

# --- moderate LSTM with Focal Loss, no early stopping, 50 epochs ---
HIDDEN=128
N_LAYERS=2
LR=3e-4
EPOCHS=50
ALPHA=0.75
GAMMA=2.0

RUN_NAME="lstm-focal-dual-h${HIDDEN}-l${N_LAYERS}-lr${LR}-a${ALPHA}-g${GAMMA}-e${EPOCHS}"
echo "Run name: ${RUN_NAME}"
nvidia-smi || true

python -m src.train_lstm_focal_dual \
    --data-dir data/lstm/clean \
    --run-name "${RUN_NAME}" \
    --out-dir runs \
    --hidden "${HIDDEN}" \
    --n-layers "${N_LAYERS}" \
    --learning-rate "${LR}" \
    --epochs "${EPOCHS}" \
    --focal-alpha "${ALPHA}" \
    --focal-gamma "${GAMMA}" \
    --device cuda

echo "Done. Sync wandb runs from a node with internet:"
echo "  wandb sync ./runs/wandb/offline-run-*"
