"""LSTM trainer for wildfire risk prediction (7-day window).

Reads cleaned parquets from `data/lstm/clean/{train,val,test}` plus
meta.joblib produced by `src.clean --mode lstm`. Each sample has up to 7
rows (one per `day_offset` 0..6, sorted oldest-first when fed to the LSTM).
Predicts `is_fire` on day_offset=0 from the 7-day sequence.

Uses the same `src.metrics.evaluate` as the XGBoost baseline so the two
runs are directly comparable on PR-AUC / F2 / recall / confusion matrix.

CALIBRATION NOTE: same caveat as baseline — train is 1:3 (pos:neg) and we
upweight positives further via `pos_weight`. Predicted probabilities are
not calibrated to real-world prevalence; threshold is tuned on val and
applied to test (both at 1:10).

Run:
    python -m src.train_lstm --data-dir data/lstm/clean --run-name lstm-v1
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


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_sequences(
    data_dir: Path,
    split: str,
    feature_cols: list[str],
    target: str,
    n_steps: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cleaned LSTM-mode parquet and reshape to (N, n_steps, n_features).

    The cleaned data has rows keyed by (sample_id, day_offset). After
    sorting by day_offset DESCENDING within each sample, the row at
    position 0 is the oldest (day_offset=6) and the row at position 6 is
    the sample day (day_offset=0). That's the natural chronological order
    the LSTM expects: oldest -> newest.

    Returns:
        X: (N, n_steps, n_features) float32
        y: (N,) int32 — the is_fire label from day_offset=0
    """
    df = pd.read_parquet(data_dir / split, engine="pyarrow")

    # Fill any missing feature columns with 0 (defensive, e.g. rare AC cats
    # absent from val/test; mirrors baseline loader behavior).
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    df = df.sort_values(["sample_id", "day_offset"], ascending=[True, False])

    # Verify each sample has exactly n_steps rows. If not we'd need padding;
    # for the current dataset the sampling notebook produces full windows.
    counts = df.groupby("sample_id", sort=False).size()
    bad = counts[counts != n_steps]
    if len(bad) > 0:
        # Drop incomplete samples loudly rather than silently mis-shaping.
        keep_ids = counts[counts == n_steps].index
        df = df[df["sample_id"].isin(keep_ids)]
        print(f"[{split}] dropped {len(bad)} samples with != {n_steps} timesteps; "
              f"kept {len(keep_ids)}")

    n_samples = df["sample_id"].nunique()
    X = df[feature_cols].to_numpy(dtype=np.float32).reshape(n_samples, n_steps, -1)

    # Pull labels from the day_offset=0 row of each sample (the prediction day).
    # We re-sort by sample_id to align with X's reshape order.
    labels = (
        df[df["day_offset"] == 0]
        .sort_values("sample_id")[target]
        .to_numpy(dtype=np.int32)
    )
    assert len(labels) == n_samples, "label count != sample count"
    return X, labels


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class FireLSTM(nn.Module):
    """Plain LSTM -> linear head. Outputs raw logits (use BCE-with-logits).

    dropout applies in two places:
      1. Between LSTM layers (via nn.LSTM's built-in dropout arg).
         Only active when n_layers > 1; ignored for single-layer LSTMs.
      2. On the final hidden state before the linear classifier (nn.Dropout).
    Default dropout=0.0 keeps the original behaviour so existing scripts
    that import FireLSTM without the argument are unaffected.
    """

    def __init__(self, n_features: int, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,  # input shape (batch, seq, features)
            # Between-layer dropout; no-op when n_layers == 1
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # Dropout on the final hidden state before the classifier
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)           # out: (batch, seq, hidden)
        last   = out[:, -1, :]          # final timestep -> (batch, hidden)
        return self.fc(self.drop(last)).squeeze(-1)  # (batch,)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Return sigmoid(logits) for the full loader, in loader order."""
    model.eval()
    scores = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="lstm")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stop after this many epochs without val PR-AUC improvement.")
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="BCEWithLogitsLoss pos_weight. If None, computed from train class balance.")
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

    print(f"[lstm] loading splits from {args.data_dir}")
    X_tr, y_tr = load_sequences(args.data_dir, "train", feature_cols, target)
    X_val, y_val = load_sequences(args.data_dir, "val", feature_cols, target)
    X_te, y_te = load_sequences(args.data_dir, "test", feature_cols, target)
    n_features = X_tr.shape[2]
    print(f"[lstm] X_train shape: {X_tr.shape}, features={n_features}")

    # pos_weight: ratio of negatives to positives in train. Used inside
    # BCEWithLogitsLoss to upweight the positive-class loss term so the
    # model doesn't ignore the (less common) fire class.
    if args.pos_weight is None:
        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        pos_weight = n_neg / max(n_pos, 1)
    else:
        pos_weight = args.pos_weight

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[lstm] device={device}")

    # --- DataLoaders ---
    # `pin_memory=True` lets PyTorch use page-locked host memory, which
    # speeds up host-to-GPU transfers. `non_blocking=True` on `.to(device)`
    # then allows that transfer to overlap with compute.
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).float())
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
    te_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).float())

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # --- Model + optimizer ---
    model = FireLSTM(n_features=n_features, hidden=args.hidden,
                     n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    # --- wandb ---
    config = vars(args).copy()
    config.update({
        "n_features": n_features,
        "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
        "train_pos_rate": float(y_tr.mean()),
        "val_pos_rate": float(y_val.mean()),
        "test_pos_rate": float(y_te.mean()),
        "pos_weight": float(pos_weight),
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

        # End-of-epoch eval on val.
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

        # Early stopping.
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

    # Restore best weights.
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final eval ---
    val_score = predict(model, val_loader, device)
    test_score = predict(model, te_loader, device)

    # Pick operating threshold on val using F2 (recall-weighted), apply to test.
    val_sweep = threshold_sweep(y_val, val_score, beta=2.0)
    op_threshold = val_sweep["best_threshold"]
    val_metrics = evaluate(y_val, val_score, beta=2.0, fixed_threshold=op_threshold)
    test_metrics = evaluate(y_te, test_score, beta=2.0, fixed_threshold=op_threshold)

    wandb.log({**{f"val/{k}": v for k, v in val_metrics.items()},
               **{f"test/{k}": v for k, v in test_metrics.items()}})

    # PR curve on val.
    val_probs_2col = np.stack([1.0 - val_score, val_score], axis=1)
    wandb.log({
        "val/pr_curve": wandb.plot.pr_curve(
            y_val, val_probs_2col, labels=["no_fire", "fire"], classes_to_plot=[1]
        ),
    })

    # Persist metrics JSON (same schema as baseline -> plot_confusion.py works).
    with open(args.out_dir / f"{args.run_name}-metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics,
                   "operating_threshold": op_threshold}, f, indent=2)

    print("val:", val_metrics)
    print("test:", test_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
