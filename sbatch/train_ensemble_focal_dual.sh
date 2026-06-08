#!/bin/bash
#SBATCH --job-name=fire-ensemble-focal-dual
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

# --- offline wandb ---
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-fire-risk}"
export WANDB_DIR="${WANDB_DIR:-./runs}"

mkdir -p logs runs

# --- config ---
HIDDEN=128
N_LAYERS=2
LR=3e-4
EPOCHS=50
ALPHA=0.75
GAMMA=2.0
XGB_WEIGHT=0.5       # 50% XGBoost + 50% LSTM in the ensemble score

# Point this at whichever baseline run you want to use as the frozen XGBoost
XGB_MODEL="runs/baseline-v1.json"

RUN_NAME="ensemble-focal-dual-h${HIDDEN}-l${N_LAYERS}-lr${LR}-a${ALPHA}-g${GAMMA}-xw${XGB_WEIGHT}-e${EPOCHS}"
echo "Run name: ${RUN_NAME}"
echo "XGBoost model: ${XGB_MODEL}"
nvidia-smi || true

python -m src.train_ensemble_focal_dual \
    --xgb-model      "${XGB_MODEL}" \
    --xgb-data-dir   data/baseline/clean \
    --lstm-data-dir  data/lstm/clean \
    --run-name       "${RUN_NAME}" \
    --out-dir        runs \
    --hidden         "${HIDDEN}" \
    --n-layers       "${N_LAYERS}" \
    --learning-rate  "${LR}" \
    --epochs         "${EPOCHS}" \
    --focal-alpha    "${ALPHA}" \
    --focal-gamma    "${GAMMA}" \
    --xgb-weight     "${XGB_WEIGHT}" \
    --device         cuda

echo "Done. Sync wandb runs:"
echo "  wandb sync ./runs/wandb/offline-run-*"
