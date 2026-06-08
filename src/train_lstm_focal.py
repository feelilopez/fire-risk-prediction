"""LSTM trainer with Focal Loss for wildfire risk prediction.

Same data pipeline / model architecture as `src.train_lstm`, but swaps
BCEWithLogitsLoss for Focal Loss. Focal Loss down-weights easy examples
and concentrates training on hard ones — a standard fix for imbalanced
detection (RetinaNet, Lin et al. 2017).

Why this for our problem:
    The LSTM sweep showed high FN counts. BCE treats every example equally
    weighted (modulo pos_weight). Focal Loss adds a `(1 - p_t)^gamma`
    multiplier so well-classified examples (high p_t) contribute much less
    to the gradient — the model spends its capacity on the borderline /
    missed cases where it currently fails.

Hyperparameters:
    alpha (default 0.75) — class balancing factor; weights the positive
        class. Plays a similar role to BCE's pos_weight but as a fraction
        in [0, 1] rather than a ratio.
    gamma (default 2.0)  — focusing exponent. gamma=0 reduces to weighted
        BCE; higher gamma punishes easy examples harder. 2.0 is the
        RetinaNet default and a safe starting point.

Run:
    python -m src.train_lstm_focal --data-dir data/lstm/clean --run-name lstm-focal-v1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

from src.metrics import evaluate, threshold_sweep
from src.train_lstm import FireLSTM, load_sequences, predict


# --------------------------------------------------------------------------
# Focal Loss
# --------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary Focal Loss operating on raw logits (numerically stable).

    Formula (per-example):
        p_t   = p     if y == 1 else 1 - p           where p = sigmoid(logit)
        a_t   = alpha if y == 1 else 1 - alpha
        loss  = -a_t * (1 - p_t)^gamma * log(p_t)

    We implement it via `binary_cross_entropy_with_logits(reduction='none')`
    to get the per-element BCE term -log(p_t) without a separate sigmoid +
    log call (which is less numerically stable).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # `targets` is float (BCE expects float). Per-example BCE = -log(p_t).
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # p_t = exp(-bce) since bce = -log(p_t).
        # Computing it this way avoids re-doing sigmoid + log explicitly.
        p_t = torch.exp(-bce)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return loss.mean()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="lstm-focal")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stop after this many epochs without val PR-AUC improvement.")
    parser.add_argument("--focal-alpha", type=float, default=0.75,
                        help="Class balancing factor in [0, 1]. Higher = positive class weighted more.")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focusing exponent. 0 = weighted BCE; 2.0 = RetinaNet default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = joblib.load(args.data_dir / "meta.joblib")
    feature_cols = meta["feature_cols"]
    target = meta["target"]

    print(f"[lstm-focal] loading splits from {args.data_dir}")
    X_tr, y_tr = load_sequences(args.data_dir, "train", feature_cols, target)
    X_val, y_val = load_sequences(args.data_dir, "val", feature_cols, target)
    X_te, y_te = load_sequences(args.data_dir, "test", feature_cols, target)
    n_features = X_tr.shape[2]
    print(f"[lstm-focal] X_train shape: {X_tr.shape}, features={n_features}")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[lstm-focal] device={device}")

    # --- DataLoaders ---
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).float())
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
    te_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).float())

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # --- Model + optimizer + Focal Loss ---
    model = FireLSTM(n_features=n_features, hidden=args.hidden,
                     n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    # --- wandb ---
    config = vars(args).copy()
    config.update({
        "n_features": n_features,
        "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
        "train_pos_rate": float(y_tr.mean()),
        "val_pos_rate": float(y_val.mean()),
        "test_pos_rate": float(y_te.mean()),
        "loss": "focal",
        "model": "FireLSTM",
    })
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "fire-risk"),
        name=args.run_name,
        config=config,
        dir=str(args.out_dir),
    )

    # --- Training loop with early stopping on val PR-AUC ---
    best_val_pr_auc = -1.0
    best_state = None
    epochs_no_improve = 0
    ckpt_path = args.out_dir / f"{args.run_name}.pt"

    for epoch in range(args.epochs):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * len(yb)
            tr_n += len(yb)

        val_scores = predict(model, val_loader, device)
        val_metrics_epoch = evaluate(y_val, val_scores, beta=2.0)
        train_loss = tr_loss_sum / max(tr_n, 1)

        log = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/pr_auc": val_metrics_epoch["pr_auc"],
            "val/roc_auc": val_metrics_epoch["roc_auc"],
            "val/recall": val_metrics_epoch["recall"],
            "val/precision": val_metrics_epoch["precision"],
            "val/f2": val_metrics_epoch["f2"],
        }
        wandb.log(log, step=epoch)
        print(f"epoch {epoch:>2}  train_loss={train_loss:.4f}  "
              f"val_pr_auc={val_metrics_epoch['pr_auc']:.4f}  "
              f"val_recall={val_metrics_epoch['recall']:.4f}")

        if val_metrics_epoch["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics_epoch["pr_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (no val PR-AUC improvement "
                      f"for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final eval ---
    val_score = predict(model, val_loader, device)
    test_score = predict(model, te_loader, device)

    val_sweep = threshold_sweep(y_val, val_score, beta=2.0)
    op_threshold = val_sweep["best_threshold"]
    val_metrics = evaluate(y_val, val_score, beta=2.0, fixed_threshold=op_threshold)
    test_metrics = evaluate(y_te, test_score, beta=2.0, fixed_threshold=op_threshold)

    wandb.log({**{f"val/{k}": v for k, v in val_metrics.items()},
               **{f"test/{k}": v for k, v in test_metrics.items()}})

    val_probs_2col = np.stack([1.0 - val_score, val_score], axis=1)
    wandb.log({
        "val/pr_curve": wandb.plot.pr_curve(
            y_val, val_probs_2col, labels=["no_fire", "fire"], classes_to_plot=[1]
        ),
    })

    with open(args.out_dir / f"{args.run_name}-metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics,
                   "operating_threshold": op_threshold}, f, indent=2)

    print("val:", val_metrics)
    print("test:", test_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
