#!/bin/bash
#SBATCH --job-name=fire-clean
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

ENV_PATH="${ENV_PATH:-./envs/fire-risk}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

mkdir -p logs

python -m src.clean --mode baseline --in-dir data/baseline --out-dir data/baseline/clean
python -m src.clean --mode lstm     --in-dir data/lstm     --out-dir data/lstm/clean

echo "Cleaning done."
