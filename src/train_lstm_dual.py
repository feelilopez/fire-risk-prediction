"""LSTM trainer (BCE) — no early stopping, tracks TWO best checkpoints.

Same architecture and data pipeline as `src.train_lstm`, but:
  - Trains for the full --epochs (default 50). No early stopping. We saw
    runs killed mid-improvement in the ablation; letting them finish.
  - Tracks the epoch with MAX val PR-AUC  -> saved as `<run>-best_pr_auc.pt`
  - Tracks the epoch with MIN val FN      -> saved as `<run>-best_fn.pt`
    (FN is taken at the F2-optimal threshold from each epoch's val sweep —
    a single threshold-independent count would be meaningless.)
  - At the end, evaluates BOTH checkpoints on val+test and writes two
    metrics JSON files using the same schema as the rest of the codebase,
    so `src.plot_confusion` works on either.

CAVEAT on "best FN": at very early epochs the model is near-random, and the
F2-optimal threshold can be low enough that FN looks artificially small.
Inspect the wandb val/pr_auc and val/fn curves together to sanity-check
that the picked epoch is in a meaningful region.

Run:
    python -m src.train_lstm_dual --data-dir data/lstm/clean --run-name lstm-dual-v1
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

from src.metrics import evaluate, threshold_sweep
from src.train_lstm import FireLSTM, load_sequences, predict


def _final_eval_and_dump(model, state_dict, val_loader, te_loader,
                         y_val, y_te, device,
                         out_dir: Path, run_name: str, tag: str,
                         meta: dict):
    """Load `state_dict` into model, evaluate on val+test, dump metrics JSON.

    `tag` is appended to filenames (e.g. "best_pr_auc", "best_fn") so each
    checkpoint gets its own metrics file with the standard schema that
    `src.plot_confusion` expects.
    """
    model.load_state_dict(state_dict)
    val_score = predict(model, val_loader, device)
    test_score = predict(model, te_loader, device)

    val_sweep = threshold_sweep(y_val, val_score, beta=2.0)
    op_threshold = val_sweep["best_threshold"]
    val_metrics = evaluate(y_val, val_score, beta=2.0, fixed_threshold=op_threshold)
    test_metrics = evaluate(y_te, test_score, beta=2.0, fixed_threshold=op_threshold)

    metrics_path = out_dir / f"{run_name}-{tag}-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics,
                   "operating_threshold": op_threshold,
                   "selection_meta": meta}, f, indent=2)

    wandb.log({**{f"{tag}/val/{k}": v for k, v in val_metrics.items()},
               **{f"{tag}/test/{k}": v for k, v in test_metrics.items()}})

    print(f"--- {tag} (epoch {meta.get('epoch')}) ---")
    print(f"  val:  pr_auc={val_metrics['pr_auc']:.4f}  recall={val_metrics['recall']:.4f}  "
          f"FN={val_metrics['fn']}  TP={val_metrics['tp']}  thr={op_threshold:.4f}")
    print(f"  test: pr_auc={test_metrics['pr_auc']:.4f}  recall={test_metrics['recall']:.4f}  "
          f"FN={test_metrics['fn']}  TP={test_metrics['tp']}")
    return val_score, val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="lstm-dual")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="BCEWithLogitsLoss pos_weight. If None, computed from train class balance.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate: applied between LSTM layers and before the classifier.")
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

    print(f"[lstm-dual] loading splits from {args.data_dir}")
    X_tr, y_tr = load_sequences(args.data_dir, "train", feature_cols, target)
    X_val, y_val = load_sequences(args.data_dir, "val", feature_cols, target)
    X_te, y_te = load_sequences(args.data_dir, "test", feature_cols, target)
    n_features = X_tr.shape[2]
    print(f"[lstm-dual] X_train shape: {X_tr.shape}, features={n_features}")

    if args.pos_weight is None:
        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        pos_weight = n_neg / max(n_pos, 1)
    else:
        pos_weight = args.pos_weight

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[lstm-dual] device={device}  pos_weight={pos_weight:.3f}")

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

    # --- Model + optimizer + loss ---
    model = FireLSTM(n_features=n_features, hidden=args.hidden,
                     n_layers=args.n_layers, dropout=args.dropout).to(device)
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
        "dropout": args.dropout,
        "loss": "bce",
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

    # --- Training loop (no early stopping, two checkpoints tracked) ---
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

        # Track best PR-AUC checkpoint.
        if val_metrics_epoch["pr_auc"] > best_pr_auc_val:
            best_pr_auc_val = val_metrics_epoch["pr_auc"]
            best_pr_auc_state = copy.deepcopy(model.state_dict())
            best_pr_auc_epoch = epoch
            torch.save(best_pr_auc_state, pr_auc_ckpt)

        # Track min FN checkpoint.
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
        # PR curve from the best-FN model for wandb.
        val_probs_2col = np.stack([1.0 - val_score, val_score], axis=1)
        wandb.log({"best_fn/val/pr_curve": wandb.plot.pr_curve(
            y_val, val_probs_2col, labels=["no_fire", "fire"], classes_to_plot=[1]
        )})

    wandb.finish()


if __name__ == "__main__":
    main()
