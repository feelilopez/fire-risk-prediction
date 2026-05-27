#!/bin/bash
#SBATCH --job-name=fire-lstm-focal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load conda
source activate fire-risk

cd /home/mfsvensson/TFG_reps/fire-risk-prediction

set -euo pipefail

# --- offline wandb (sync later from a login node with internet) ---
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-fire-risk}"
export WANDB_DIR="${WANDB_DIR:-./runs}"

mkdir -p logs runs

# --- moderate LSTM: hidden=128, n_layers=2, Focal Loss (alpha=0.75, gamma=2.0) ---
HIDDEN=128
N_LAYERS=2
LR=3e-4
ALPHA=0.75
GAMMA=2.0

RUN_NAME="lstm-focal-h${HIDDEN}-l${N_LAYERS}-lr${LR}-a${ALPHA}-g${GAMMA}"
echo "Run name: ${RUN_NAME}"
nvidia-smi || true

python -m src.train_lstm_focal \
    --data-dir data/lstm/clean \
    --run-name "${RUN_NAME}" \
    --out-dir runs \
    --hidden "${HIDDEN}" \
    --n-layers "${N_LAYERS}" \
    --learning-rate "${LR}" \
    --focal-alpha "${ALPHA}" \
    --focal-gamma "${GAMMA}" \
    --device cuda

echo "Done. Sync wandb runs from a node with internet:"
echo "  wandb sync ./runs/wandb/offline-run-*"
