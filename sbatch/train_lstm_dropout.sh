#!/bin/bash
#SBATCH --job-name=fire-lstm-dropout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=09:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load conda
source activate fire-risk

cd /home/mfsvensson/TFG_reps/fire-risk-prediction

set -euo pipefail

export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-fire-risk}"
export WANDB_DIR="${WANDB_DIR:-./runs}"

mkdir -p logs runs

# Fixed config for all three runs
HIDDEN=128
N_LAYERS=2
LR=1e-4
EPOCHS=50

# --- Run 1: dropout=0.2 ---
DROPOUT=0.2
RUN_NAME="last_test_${DROPOUT}"
echo "========================================"
echo "Run: ${RUN_NAME}"
echo "========================================"
nvidia-smi || true

python -m src.train_lstm_dual \
    --data-dir      data/lstm/clean \
    --run-name      "${RUN_NAME}" \
    --out-dir       runs \
    --hidden        "${HIDDEN}" \
    --n-layers      "${N_LAYERS}" \
    --learning-rate "${LR}" \
    --epochs        "${EPOCHS}" \
    --dropout       "${DROPOUT}" \
    --device        cuda

# --- Run 2: dropout=0.3 ---
DROPOUT=0.3
RUN_NAME="last_test_${DROPOUT}"
echo "========================================"
echo "Run: ${RUN_NAME}"
echo "========================================"

python -m src.train_lstm_dual \
    --data-dir      data/lstm/clean \
    --run-name      "${RUN_NAME}" \
    --out-dir       runs \
    --hidden        "${HIDDEN}" \
    --n-layers      "${N_LAYERS}" \
    --learning-rate "${LR}" \
    --epochs        "${EPOCHS}" \
    --dropout       "${DROPOUT}" \
    --device        cuda

# --- Run 3: dropout=0.35 ---
DROPOUT=0.35
RUN_NAME="last_test_${DROPOUT}"
echo "========================================"
echo "Run: ${RUN_NAME}"
echo "========================================"

python -m src.train_lstm_dual \
    --data-dir      data/lstm/clean \
    --run-name      "${RUN_NAME}" \
    --out-dir       runs \
    --hidden        "${HIDDEN}" \
    --n-layers      "${N_LAYERS}" \
    --learning-rate "${LR}" \
    --epochs        "${EPOCHS}" \
    --dropout       "${DROPOUT}" \
    --device        cuda

echo "========================================"
echo "All 3 runs done."
echo "Sync wandb: wandb sync ./runs/wandb/offline-run-*"
echo "========================================"
