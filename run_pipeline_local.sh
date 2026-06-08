#!/usr/bin/env bash
# Run the local verification pipeline for wildfire risk prediction.
# This script ensures that all code phases (cleaning, training, evaluation)
# are runnable and reproducible locally on CPU with small epoch counts

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# 1. Disable wandb sync to allow running offline without credentials
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# Create runs directory if it doesn't exist
mkdir -p runs

echo "=== 1. Starting Data Cleaning ==="
# Clean baseline data
python -m src.clean --mode baseline --in-dir data/baseline --out-dir data/baseline/clean
# Clean LSTM data
python -m src.clean --mode lstm --in-dir data/lstm --out-dir data/lstm/clean

echo "=== 2. Training Baseline XGBoost Model (CPU, fast run) ==="
python -m src.train_baseline \
    --data-dir data/baseline/clean \
    --run-name baseline-local-test \
    --out-dir runs \
    --n-estimators 10 \
    --device cpu

echo "=== 3. Training LSTM Model (CPU, fast run) ==="
python -m src.train_lstm \
    --data-dir data/lstm/clean \
    --run-name lstm-local-test \
    --out-dir runs \
    --epochs 2 \
    --num-workers 0 \
    --device cpu

echo "=== 4. Evaluating Baseline XGBoost Model ==="
python -m src.eval_threshold \
    --checkpoint runs/baseline-local-test.json \
    --data-dir data/baseline/clean \
    --beta 2.0 \
    --device cpu

echo "=== 5. Evaluating LSTM Model ==="
python -m src.eval_threshold \
    --checkpoint runs/lstm-local-test.pt \
    --data-dir data/lstm/clean \
    --beta 2.0 \
    --device cpu

echo "=== Pipeline check completed successfully! ==="
