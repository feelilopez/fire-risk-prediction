#!/bin/bash
#SBATCH --job-name=fire-lstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
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

# --- ablation grid: 3 hidden * 3 layers * 3 lr = 27 combos, sequential ---
HIDDENS=(32 64 128)
LAYERS=(1 2 3)
LRS=(1e-3 3e-4 1e-4)

nvidia-smi || true

for HIDDEN in "${HIDDENS[@]}"; do
    for N_LAYERS in "${LAYERS[@]}"; do
        for LR in "${LRS[@]}"; do
            RUN_NAME="lstm-h${HIDDEN}-l${N_LAYERS}-lr${LR}"
            echo "==================================================="
            echo "Starting ${RUN_NAME}  ($(date))"
            echo "==================================================="

            python -m src.train_lstm \
                --data-dir data/lstm/clean \
                --run-name "${RUN_NAME}" \
                --out-dir runs \
                --hidden "${HIDDEN}" \
                --n-layers "${N_LAYERS}" \
                --learning-rate "${LR}" \
                --device cuda

            echo "Finished ${RUN_NAME}  ($(date))"
        done
    done
done

echo "All 27 runs done. Sync wandb from a node with internet:"
echo "  wandb sync ./runs/wandb/offline-run-*"
