#!/bin/bash
#SBATCH --job-name=fire-lstm-dual
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

# --- moderate LSTM with BCE loss, no early stopping, 50 epochs ---
HIDDEN=128
N_LAYERS=2
LR=1e-3
EPOCHS=50

RUN_NAME="lstm-dual-h${HIDDEN}-l${N_LAYERS}-lr${LR}-e${EPOCHS}"
echo "Run name: ${RUN_NAME}"
nvidia-smi || true

python -m src.train_lstm_dual \
    --data-dir data/lstm/clean \
    --run-name "${RUN_NAME}" \
    --out-dir runs \
    --hidden "${HIDDEN}" \
    --n-layers "${N_LAYERS}" \
    --learning-rate "${LR}" \
    --epochs "${EPOCHS}" \
    --device cuda

echo "Done. Sync wandb runs from a node with internet:"
echo "  wandb sync ./runs/wandb/offline-run-*"
