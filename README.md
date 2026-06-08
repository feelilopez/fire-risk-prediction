# Wildfire Risk Prediction in Spain

This repository implements a machine learning and deep learning pipeline to predict wildfire risk in mainland Spain using the spatio-temporal [IberFire dataset](https://arxiv.org/abs/2505.00837). We focus on predicting the probability of fire occurrence for the next day based on the previous week's data.

The project addresses the extreme class imbalance of fire events (~0.002% of observations) using spatial and temporal hard negatives, and evaluates models using recall-focused metrics like PR-AUC score to minimize missed fire events (false negatives).

For details on the project methodology, models, and results, please refer to [report.md](report.md).

---

## Repository Structure

```text
├── 1-EDA.ipynb                 # Exploratory Data Analysis
├── 2-sampling.ipynb            # Train (1:3 balanced), val, and test (1:10) sampling
├── report.md                   # Detailed project report
├── results/                    # Obtained evaluation metrics and plots
├── run_pipeline_local.sh       # Local verification pipeline (runs on CPU)

├── requirements.txt            # Python package dependencies (pip setup)
├── environment-local.yml       # Local Conda environment definition (CPU-only)
├── environment-cluster.yml     # Cluster Conda environment definition (GPU / CUDA 12.1)
├── src/                        # Core Python scripts
│   ├── clean.py                # Data imputation, normalization, and feature selection
│   ├── train_baseline.py       # XGBoost baseline training
│   ├── train_lstm.py           # PyTorch LSTM network training (7-day temporal window)
│   ├── eval_threshold.py       # Operating threshold tuning and confusion matrix plotting
│   ├── metrics.py              # Custom metrics (PR-AUC, F-beta sweep, recall)
│   └── plot_confusion.py       # Script for confusion matrix plotting
├── sbatch/                     # Slurm cluster submission scripts
│   ├── clean.sh
│   ├── train_baseline.sh
│   └── train_lstm*.sh
└── data/                       # Dataset directories
    ├── baseline/               # Baseline XGBoost training data
    └── lstm/                   # LSTM training data
```

---

## Environment Setup

You can set up the project environment using either **Conda** (recommended) or **pip**.

### Option A: Conda (Recommended)

* **For local work (notebooks, cleaning, lightweight testing):**
  ```bash
  conda env create -f environment-local.yml
  conda activate fire-risk-local
  ```

* **For cluster GPU training (PyTorch with CUDA, GPU XGBoost, wandb):**
  ```bash
  conda env create -f environment-cluster.yml
  conda activate fire-risk
  ```

### Option B: pip

Create a virtual environment and install the package requirements:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Execution Pipeline

The workflow consists of five stages: Exploratory Data Analysis, Data Sampling, Data Cleaning, Model Training, and Model Evaluation.

### 1. Exploratory Data Analysis (`1-EDA.ipynb`)
Run this notebook to explore the IberFire spatio-temporal dataset, inspect the target variable `is_fire` across Spain, and visualize features.

### 2. Data Sampling (`2-sampling.ipynb`)
Run this notebook to generate the train, validation, and test datasets:
* **Train split (2008-2022):** Sampled with a 1:3 ratio of fire to non-fire cells (using spatial hard negatives, temporal hard negatives, and baseline negatives).
* **Validation (2023) and Test (2024) splits:** Sampled at a 1:10 ratio of fire to non-fire cells to reflect realistic imbalanced evaluations.
* **Outputs:** Writes datasets to `data/baseline/` (flat daily samples) and `data/lstm/` (7-day window sequences).

Note: this notebook can take quite long to run in standard CPUs. It is recommended to use the sampled datasets provided in `/data` instead.

### 3. Data Cleaning (`src/clean.py`)
Cleans the sampled Parquets by removing zero-variance columns, collapsing yearly CORINE Land Cover (CLC) and population density into single columns, applying temporal forward-fill (for LSTM), imputing missing values, and scaling features.

* **To run locally:**
  ```bash
  python -m src.clean --mode baseline --in-dir data/baseline --out-dir data/baseline/clean
  python -m src.clean --mode lstm --in-dir data/lstm --out-dir data/lstm/clean
  ```
* **To run on Slurm cluster:**
  ```bash
  sbatch sbatch/clean.sh
  ```

### 4. Model Training
Train models using either the XGBoost baseline or PyTorch LSTM scripts in `src/`.

* **XGBoost Baseline:**
  * Local (CPU):
    ```bash
    python -m src.train_baseline --data-dir data/baseline/clean --device cpu --n-estimators 500
    ```
  * Cluster (GPU):
    ```bash
    sbatch sbatch/train_baseline.sh
    ```

* **LSTM Network:**
  * Local (CPU):
    ```bash
    python -m src.train_lstm --data-dir data/lstm/clean --device cpu --epochs 5 --num-workers 0
    ```
  * Cluster (GPU):
    ```bash
    sbatch sbatch/train_lstm.sh
    ```

### 5. Model Evaluation (`src/eval_threshold.py`)
Evaluate trained checkpoints, sweep classification thresholds, and compute test metrics. It also plots the Precision-Recall threshold sweep and the confusion matrices.

* **Evaluate XGBoost Baseline:**
  ```bash
  python -m src.eval_threshold --checkpoint runs/baseline.json --data-dir data/baseline/clean --beta 2.0
  ```

* **Evaluate LSTM:**
  ```bash
  python -m src.eval_threshold --checkpoint runs/lstm.pt --data-dir data/lstm/clean --beta 2.0
  ```

---

## Quick Verification (Local CPU Run)

We provide a master bash script to quickly verify that the cleaning, training, and evaluation scripts work end-to-end on CPU without requiring an active Weights & Biases configuration:

```bash
bash run_pipeline_local.sh
```

This runs a lightweight training (10 estimators for XGBoost, 2 epochs for LSTM) and verifies that metrics and confusion matrix plots are successfully generated in the `runs/` directory.
