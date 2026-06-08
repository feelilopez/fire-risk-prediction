"""Compute, plot, and compare feature importance for XGBoost and LSTM models.

1. XGBoost: Native Gain-based importance.
2. LSTM & XGBoost: Permutation Feature Importance on the test or validation split
   (shuffling each feature across the batch to measure the drop in PR-AUC).
3. Plotting: Saves clean, publication-ready horizontal bar charts for all three methods.

Usage:
    python -m src.feature_importance \\
        --xgb-model new_runs/baseline-20260526-190623.json \\
        --lstm-model new_runs/lstm-focal-dual-h128-l2-lr3e-4-a0.75-g2.0-e50-best_pr_auc.pt \\
        --split test
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
import seaborn as sns

from src.train_lstm import FireLSTM, load_sequences
from src.train_baseline import load_xy, _sanitize_feature_names

# XGBoost bad char pattern
_XGB_BAD = re.compile(r"[\[\]<]")

def sanitize_name(n: str) -> str:
    return _XGB_BAD.sub("_", n)

@torch.no_grad()
def evaluate_lstm(model, loader, device) -> float:
    """Compute validation/test PR-AUC for LSTM."""
    model.eval()
    all_scores = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        scores = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(scores)
        all_targets.append(yb.numpy())
    
    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_targets)
    return float(average_precision_score(y_true, y_score))

def evaluate_xgb(booster, X, y) -> float:
    """Compute validation/test PR-AUC for XGBoost."""
    dmat = xgb.DMatrix(X)
    y_score = booster.predict(dmat)
    return float(average_precision_score(y, y_score))

def plot_importance(importance_list, title, xlabel, save_path, palette_name="Oranges_r"):
    """Generate and save a horizontal bar plot of feature importances."""
    if not importance_list:
        return
        
    df = pd.DataFrame(importance_list[:15], columns=["Feature", "Importance"])
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot using seaborn
    sns.barplot(
        x="Importance", 
        y="Feature", 
        data=df, 
        hue="Feature", 
        legend=False, 
        palette=sns.color_palette(palette_name, n_colors=15)
    )
    
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-data-dir", type=Path, default=Path("data/baseline/clean"),
                        help="Path to baseline cleaned data directory.")
    parser.add_argument("--lstm-data-dir", type=Path, default=Path("data/lstm/clean"),
                        help="Path to LSTM cleaned data directory.")
    parser.add_argument("--xgb-model", type=Path, default=Path("new_runs/baseline-20260526-190623.json"),
                        help="Path to trained XGBoost booster .json model.")
    parser.add_argument("--lstm-model", type=Path, default=Path("new_runs/lstm-focal-dual-h128-l2-lr3e-4-a0.75-g2.0-e50-best_pr_auc.pt"),
                        help="Path to trained LSTM state_dict .pt model.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Which data split to evaluate on (val or test).")
    parser.add_argument("--hidden", type=int, default=128, help="LSTM hidden state dimension.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for LSTM eval.")
    parser.add_argument("--out-dir", type=Path, default=Path("new_runs"), help="Output directory for plots and JSON.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Running feature importance analysis on the '{args.split}' split...\n")

    # ==========================================================================
    # 1. XGBoost Native Feature Importance (Gain)
    # ==========================================================================
    print("--- 1. XGBoost Native Feature Importance (Gain) ---")
    if not args.xgb_model.exists():
        print(f"XGBoost model file {args.xgb_model} not found. Skipping.")
        xgb_gain_sorted = []
    else:
        # Load meta to map sanitized names back to raw names
        meta_xgb = joblib.load(args.xgb_data_dir / "meta.joblib")
        xgb_feature_cols = meta_xgb["feature_cols"]
        san_to_raw = {sanitize_name(name): name for name in xgb_feature_cols}

        booster = xgb.Booster()
        booster.load_model(args.xgb_model)
        
        # XGBoost importance (Gain)
        xgb_gain = booster.get_score(importance_type="gain")
        
        # Map back to raw names
        mapped_gain = {}
        for san_name, score in xgb_gain.items():
            raw_name = san_to_raw.get(san_name, san_name)
            mapped_gain[raw_name] = float(score)

        # Sort and print
        xgb_gain_sorted = sorted(mapped_gain.items(), key=lambda kv: kv[1], reverse=True)
        print("Top 15 XGBoost Features by Gain:")
        for rank, (feat, score) in enumerate(xgb_gain_sorted[:15], 1):
            print(f"  {rank:>2}. {feat:<40} : {score:.4f}")
        
        # Plot
        plot_importance(
            xgb_gain_sorted,
            title="XGBoost Feature Importance (Gain)",
            xlabel="Average Gain (Loss Reduction Contribution)",
            save_path=args.out_dir / "xgb_importance_gain.png",
            palette_name="Oranges_r"
        )

    # ==========================================================================
    # 2. Permutation Feature Importance for LSTM
    # ==========================================================================
    print(f"\n--- 2. LSTM Permutation Feature Importance (PR-AUC Drop on {args.split}) ---")
    if not args.lstm_model.exists():
        print(f"LSTM model file {args.lstm_model} not found. Skipping.")
        lstm_perm_sorted = []
    else:
        # Load LSTM meta
        meta_lstm = joblib.load(args.lstm_data_dir / "meta.joblib")
        lstm_feature_cols = meta_lstm["feature_cols"]
        n_features = len(lstm_feature_cols)
        
        # Load model
        model = FireLSTM(n_features=n_features, hidden=args.hidden, n_layers=args.n_layers)
        state_dict = torch.load(args.lstm_model, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        
        # Load dataset split
        X_split_lstm, y_split_lstm = load_sequences(args.lstm_data_dir, args.split, lstm_feature_cols, meta_lstm["target"])
        split_ds = TensorDataset(
            torch.from_numpy(X_split_lstm.copy()), 
            torch.from_numpy(y_split_lstm.copy()).float()
        )
        split_loader = DataLoader(split_ds, batch_size=args.batch_size, shuffle=False)
        
        # Base score
        base_pr_auc_lstm = evaluate_lstm(model, split_loader, device)
        print(f"Base {args.split.capitalize()} PR-AUC for LSTM: {base_pr_auc_lstm:.4f}")

        # Compute drops
        lstm_perm_scores = {}
        for idx, feat_name in enumerate(lstm_feature_cols):
            # Permute values of this feature across samples (batch dimension)
            # Input shape: (N, seq_len, n_features)
            X_perm = X_split_lstm.copy()
            # Shuffle along axis 0 (batch size) for this specific feature index across all timesteps
            perm_indices = np.random.permutation(len(X_perm))
            X_perm[:, :, idx] = X_perm[perm_indices, :, idx]
            
            perm_ds = TensorDataset(
                torch.from_numpy(X_perm.copy()), 
                torch.from_numpy(y_split_lstm.copy()).float()
            )
            perm_loader = DataLoader(perm_ds, batch_size=args.batch_size, shuffle=False)
            
            perm_pr_auc = evaluate_lstm(model, perm_loader, device)
            drop = base_pr_auc_lstm - perm_pr_auc
            lstm_perm_scores[feat_name] = float(drop)
            
        lstm_perm_sorted = sorted(lstm_perm_scores.items(), key=lambda kv: kv[1], reverse=True)
        print(f"Top 15 LSTM Features by Permutation PR-AUC Drop ({args.split}):")
        for rank, (feat, score) in enumerate(lstm_perm_sorted[:15], 1):
            print(f"  {rank:>2}. {feat:<40} : {score:.6f}")
            
        # Plot
        plot_importance(
            lstm_perm_sorted,
            title=f"LSTM Permutation Importance (PR-AUC Drop on {args.split.capitalize()})",
            xlabel="Validation/Test PR-AUC Drop (Shuffled vs. Baseline)",
            save_path=args.out_dir / "lstm_importance_permutation.png",
            palette_name="Blues_r"
        )

    # ==========================================================================
    # 3. Save results
    # ==========================================================================
    results = {
        "xgb_gain": xgb_gain_sorted,
        "lstm_permutation_drop": lstm_perm_sorted,
    }
    
    out_file = args.out_dir / f"feature_importance_results_{args.split}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON saved to {out_file}")

if __name__ == "__main__":
    main()
