"""LSTM trainer (Focal Loss) — no early stopping, tracks TWO best checkpoints.

Identical to `src.train_lstm_dual` except the loss is Focal instead of BCE.
See that file's docstring for the full design.

Run:
    python -m src.train_lstm_focal_dual --data-dir data/lstm/clean --run-name lstm-focal-dual-v1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb

from src.metrics import evaluate, threshold_sweep
from src.train_lstm import FireLSTM, load_sequences, predict
from src.train_lstm_focal import FocalLoss
from src.train_lstm_dual import _final_eval_and_dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="lstm-focal-dual")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
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

    print(f"[lstm-focal-dual] loading splits from {args.data_dir}")
    X_tr, y_tr = load_sequences(args.data_dir, "train", feature_cols, target)
    X_val, y_val = load_sequences(args.data_dir, "val", feature_cols, target)
    X_te, y_te = load_sequences(args.data_dir, "test", feature_cols, target)
    n_features = X_tr.shape[2]
    print(f"[lstm-focal-dual] X_train shape: {X_tr.shape}, features={n_features}")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[lstm-focal-dual] device={device}  alpha={args.focal_alpha}  gamma={args.focal_gamma}")

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
        "early_stopping": False,
    })
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "fire-risk"),
        name=args.run_name,
        config=config,
        dir=str(args.out_dir),
    )

    # --- Training loop ---
    best_pr_auc_val = -1.0
    best_pr_auc_state = None
    best_pr_auc_epoch = -1

    min_fn_val = float("inf")
    best_fn_state = None
    best_fn_epoch = -1
    best_fn_threshold = None

    pr_auc_ckpt = args.out_dir / f"{args.run_name}-best_pr_auc.pt"
    fn_ckpt = args.out_dir / f"{args.run_name}-best_fn.pt"

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

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/pr_auc": val_metrics_epoch["pr_auc"],
            "val/roc_auc": val_metrics_epoch["roc_auc"],
            "val/recall": val_metrics_epoch["recall"],
            "val/precision": val_metrics_epoch["precision"],
            "val/f2": val_metrics_epoch["f2"],
            "val/fn": val_metrics_epoch["fn"],
            "val/tp": val_metrics_epoch["tp"],
            "val/threshold": val_metrics_epoch["threshold"],
        }, step=epoch)

        print(f"epoch {epoch:>2}  train_loss={train_loss:.4f}  "
              f"val_pr_auc={val_metrics_epoch['pr_auc']:.4f}  "
              f"val_recall={val_metrics_epoch['recall']:.4f}  "
              f"val_FN={val_metrics_epoch['fn']}")

        if val_metrics_epoch["pr_auc"] > best_pr_auc_val:
            best_pr_auc_val = val_metrics_epoch["pr_auc"]
            best_pr_auc_state = copy.deepcopy(model.state_dict())
            best_pr_auc_epoch = epoch
            torch.save(best_pr_auc_state, pr_auc_ckpt)

        if val_metrics_epoch["fn"] < min_fn_val:
            min_fn_val = val_metrics_epoch["fn"]
            best_fn_state = copy.deepcopy(model.state_dict())
            best_fn_epoch = epoch
            best_fn_threshold = val_metrics_epoch["threshold"]
            torch.save(best_fn_state, fn_ckpt)

    # --- Final eval for both checkpoints ---
    print()
    print("=" * 60)
    print(f"Training done. Evaluating two best checkpoints.")
    print("=" * 60)

    if best_pr_auc_state is not None:
        _final_eval_and_dump(
            model, best_pr_auc_state, val_loader, te_loader, y_val, y_te, device,
            args.out_dir, args.run_name, tag="best_pr_auc",
            meta={"epoch": best_pr_auc_epoch, "criterion": "max val PR-AUC",
                  "val_pr_auc_at_pick": best_pr_auc_val},
        )

    if best_fn_state is not None:
        val_score, _, _ = _final_eval_and_dump(
            model, best_fn_state, val_loader, te_loader, y_val, y_te, device,
            args.out_dir, args.run_name, tag="best_fn",
            meta={"epoch": best_fn_epoch, "criterion": "min val FN at F2-optimal threshold",
                  "val_fn_at_pick": int(min_fn_val),
                  "threshold_at_pick": float(best_fn_threshold) if best_fn_threshold is not None else None},
        )
        val_probs_2col = np.stack([1.0 - val_score, val_score], axis=1)
        wandb.log({"best_fn/val/pr_curve": wandb.plot.pr_curve(
            y_val, val_probs_2col, labels=["no_fire", "fire"], classes_to_plot=[1]
        )})

    wandb.finish()


if __name__ == "__main__":
    main()
