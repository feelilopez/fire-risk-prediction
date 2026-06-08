"""Ensemble trainer: frozen XGBoost + LSTM with Focal Loss.

Design
------
The XGBoost model is loaded once and kept frozen throughout training.
Only the LSTM is trained (with Focal Loss, no early stopping, 50 epochs).

At the end of every epoch we compute:

    ensemble_score = xgb_weight * xgb_val_score
                   + (1 - xgb_weight) * lstm_val_score

and use that COMBINED score to decide which LSTM checkpoint to keep:
  - best_pr_auc → highest ensemble PR-AUC on val
  - best_fn     → lowest ensemble FN on val (at F2-optimal threshold)

The XGBoost scores are pre-computed ONCE at the start from the baseline
cleaned data and never change — XGB sees different-format features (flat,
1-row per sample) while the LSTM sees 7-day sequences.

Alignment requirement
---------------------
Both data dirs must contain splits with the SAME number of rows and the
SAME label distribution. The script verifies this at startup. If they don't
match you'll get an assertion error — check that both were cleaned from the
same source split parquets.

Run:
    python -m src.train_ensemble_focal_dual \\
        --xgb-model   runs/baseline-v1.json \\
        --xgb-data-dir data/baseline/clean \\
        --lstm-data-dir data/lstm/clean \\
        --run-name ensemble-focal-dual-v1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb

from src.metrics import evaluate, threshold_sweep
from src.train_lstm import FireLSTM, load_sequences, predict
from src.train_lstm_focal import FocalLoss


# ---------------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------------

_XGB_BAD = re.compile(r"[\[\]<]")

def _sanitize(names: list[str]) -> list[str]:
    return [_XGB_BAD.sub("_", n) for n in names]


def _xgb_predict_from_flat(booster, data_dir: Path, split: str,
                            feature_cols: list[str], target: str):
    """Load a flat baseline parquet and return (scores, y) using the booster."""
    import pandas as pd
    import xgboost as xgb

    df = pd.read_parquet(data_dir / split, engine="pyarrow")
    for c in feature_cols:          # fill rare categories missing from val/test
        if c not in df.columns:
            df[c] = 0.0
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.int32)
    dm = xgb.DMatrix(X, feature_names=_sanitize(feature_cols))
    return booster.predict(dm), y


# ---------------------------------------------------------------------------
# Final evaluation helper (ensemble-aware)
# ---------------------------------------------------------------------------

def _ensemble_final_eval_and_dump(
    model, state_dict,
    val_loader, te_loader,
    y_val, y_te,
    xgb_val_score, xgb_test_score,
    xgb_weight: float,
    device: str,
    out_dir: Path, run_name: str, tag: str,
    meta: dict,
):
    """Load checkpoint, compute ensemble scores, evaluate on val+test, dump JSON.

    Returns (ensemble_val_score, val_metrics, test_metrics).
    """
    # Reload LSTM weights for this checkpoint
    model.load_state_dict(state_dict)
    lstm_val_score  = predict(model, val_loader,  device)
    lstm_test_score = predict(model, te_loader,   device)

    # Combine with frozen XGBoost scores
    lstm_weight = 1.0 - xgb_weight
    ens_val  = xgb_weight * xgb_val_score  + lstm_weight * lstm_val_score
    ens_test = xgb_weight * xgb_test_score + lstm_weight * lstm_test_score

    # Threshold is chosen on val ensemble score, applied to test ensemble score
    val_sweep    = threshold_sweep(y_val, ens_val, beta=2.0)
    op_threshold = val_sweep["best_threshold"]
    val_metrics  = evaluate(y_val, ens_val,  beta=2.0, fixed_threshold=op_threshold)
    test_metrics = evaluate(y_te,  ens_test, beta=2.0, fixed_threshold=op_threshold)

    metrics_path = out_dir / f"{run_name}-{tag}-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "val": val_metrics, "test": test_metrics,
            "operating_threshold": op_threshold,
            "selection_meta": meta,
            "xgb_weight": xgb_weight,
        }, f, indent=2)

    wandb.log({
        **{f"{tag}/val/{k}":  v for k, v in val_metrics.items()},
        **{f"{tag}/test/{k}": v for k, v in test_metrics.items()},
    })

    print(f"--- {tag} (epoch {meta.get('epoch')}) ---")
    print(f"  val:  pr_auc={val_metrics['pr_auc']:.4f}  "
          f"recall={val_metrics['recall']:.4f}  "
          f"FN={val_metrics['fn']}  TP={val_metrics['tp']}  "
          f"thr={op_threshold:.4f}")
    print(f"  test: pr_auc={test_metrics['pr_auc']:.4f}  "
          f"recall={test_metrics['recall']:.4f}  "
          f"FN={test_metrics['fn']}  TP={test_metrics['tp']}")

    return ens_val, val_metrics, test_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Required paths
    parser.add_argument("--xgb-model",    type=Path, required=True,
                        help="Path to trained XGBoost booster .json file.")
    parser.add_argument("--xgb-data-dir", type=Path, required=True,
                        help="data/baseline/clean — flat parquets for XGBoost inference.")
    parser.add_argument("--lstm-data-dir", type=Path, required=True,
                        help="data/lstm/clean — sequence parquets for LSTM training.")
    # Run config
    parser.add_argument("--run-name",  type=str, default="ensemble-focal-dual")
    parser.add_argument("--out-dir",   type=Path, default=Path("runs"))
    # LSTM architecture
    parser.add_argument("--hidden",    type=int, default=128)
    parser.add_argument("--n-layers",  type=int, default=2)
    # Training
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    # Focal loss
    parser.add_argument("--focal-alpha", type=float, default=0.75,
                        help="Class balance factor in [0,1]. Higher = more weight on positives.")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focusing exponent. 0 = weighted BCE; 2.0 = RetinaNet default.")
    # Ensemble mix
    parser.add_argument("--xgb-weight", type=float, default=0.5,
                        help="Weight for XGBoost in the ensemble (LSTM gets 1 - this).")
    # Misc
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load LSTM data (sequences)
    # -----------------------------------------------------------------------
    lstm_meta    = joblib.load(args.lstm_data_dir / "meta.joblib")
    lstm_feat    = lstm_meta["feature_cols"]
    target       = lstm_meta["target"]

    print(f"[ensemble] loading LSTM sequences from {args.lstm_data_dir}")
    X_tr,  y_tr  = load_sequences(args.lstm_data_dir, "train", lstm_feat, target)
    X_val, y_val = load_sequences(args.lstm_data_dir, "val",   lstm_feat, target)
    X_te,  y_te  = load_sequences(args.lstm_data_dir, "test",  lstm_feat, target)
    n_features   = X_tr.shape[2]
    print(f"[ensemble] LSTM  train={X_tr.shape}  val={X_val.shape}  test={X_te.shape}")

    # -----------------------------------------------------------------------
    # 2. Load XGBoost model and pre-compute frozen val/test scores
    # -----------------------------------------------------------------------
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required. Install it in the cluster env.")

    xgb_meta = joblib.load(args.xgb_data_dir / "meta.joblib")
    xgb_feat = xgb_meta["feature_cols"]

    booster = xgb.Booster()
    booster.load_model(str(args.xgb_model))
    print(f"[ensemble] loaded XGBoost booster from {args.xgb_model}")

    print(f"[ensemble] computing frozen XGB scores on {args.xgb_data_dir} …")
    xgb_val_score,  y_xgb_val  = _xgb_predict_from_flat(booster, args.xgb_data_dir, "val",  xgb_feat, target)
    xgb_test_score, y_xgb_test = _xgb_predict_from_flat(booster, args.xgb_data_dir, "test", xgb_feat, target)

    # -----------------------------------------------------------------------
    # 3. Verify alignment: LSTM and XGB splits must have same N and same y
    # -----------------------------------------------------------------------
    for split_name, y_lstm, y_xgb in [
        ("val",  y_val, y_xgb_val),
        ("test", y_te,  y_xgb_test),
    ]:
        assert len(y_lstm) == len(y_xgb), (
            f"[{split_name}] LSTM N={len(y_lstm)} != XGB N={len(y_xgb)}. "
            f"Check that both data dirs were cleaned from the same source splits."
        )
        assert abs(y_lstm.sum() - y_xgb.sum()) <= max(5, 0.01 * y_lstm.sum()), (
            f"[{split_name}] positive counts differ: LSTM={y_lstm.sum()} XGB={y_xgb.sum()}. "
            f"Splits may not be aligned."
        )
    print(f"[ensemble] alignment OK — val N={len(y_val):,}  test N={len(y_te):,}")

    # -----------------------------------------------------------------------
    # 4. Device + DataLoaders
    # -----------------------------------------------------------------------
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[ensemble] device={device}  xgb_weight={args.xgb_weight}  "
          f"focal_alpha={args.focal_alpha}  focal_gamma={args.focal_gamma}")

    def _loader(X, y, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=(device == "cuda"))

    tr_loader  = _loader(X_tr,  y_tr.astype(np.float32),  shuffle=True)
    val_loader = _loader(X_val, y_val.astype(np.float32))
    te_loader  = _loader(X_te,  y_te.astype(np.float32))

    # -----------------------------------------------------------------------
    # 5. LSTM model + optimizer + Focal Loss
    # -----------------------------------------------------------------------
    model     = FireLSTM(n_features=n_features, hidden=args.hidden,
                         n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn   = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    # -----------------------------------------------------------------------
    # 6. wandb
    # -----------------------------------------------------------------------
    config = vars(args).copy()
    config.update({
        "n_features": n_features,
        "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
        "train_pos_rate": float(y_tr.mean()),
        "val_pos_rate":   float(y_val.mean()),
        "test_pos_rate":  float(y_te.mean()),
        "loss": "focal", "model": "ensemble_FireLSTM+XGB",
        "early_stopping": False,
    })
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "fire-risk"),
        name=args.run_name,
        config=config,
        dir=str(args.out_dir),
    )

    # -----------------------------------------------------------------------
    # 7. Training loop — checkpoint on ENSEMBLE val metrics
    # -----------------------------------------------------------------------
    best_pr_auc_val = -1.0
    best_pr_auc_state, best_pr_auc_epoch = None, -1

    min_fn_val = float("inf")
    best_fn_state, best_fn_epoch, best_fn_threshold = None, -1, None

    pr_auc_ckpt = args.out_dir / f"{args.run_name}-best_pr_auc.pt"
    fn_ckpt     = args.out_dir / f"{args.run_name}-best_fn.pt"

    lstm_w = 1.0 - args.xgb_weight

    for epoch in range(args.epochs):
        # --- Train LSTM for one epoch ---
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * len(yb)
            tr_n        += len(yb)

        # --- Ensemble val score ---
        lstm_val_score  = predict(model, val_loader, device)
        ens_val_score   = args.xgb_weight * xgb_val_score + lstm_w * lstm_val_score
        ens_val_metrics = evaluate(y_val, ens_val_score, beta=2.0)
        # Also log LSTM-only metrics so we can see each component's contribution
        lstm_val_metrics = evaluate(y_val, lstm_val_score, beta=2.0)
        train_loss = tr_loss_sum / max(tr_n, 1)

        wandb.log({
            "epoch":              epoch,
            "train/loss":         train_loss,
            # Ensemble metrics (what drives checkpoint selection)
            "ens_val/pr_auc":    ens_val_metrics["pr_auc"],
            "ens_val/recall":    ens_val_metrics["recall"],
            "ens_val/fn":        ens_val_metrics["fn"],
            "ens_val/f2":        ens_val_metrics["f2"],
            "ens_val/threshold": ens_val_metrics["threshold"],
            # LSTM-only metrics (diagnostic)
            "lstm_val/pr_auc":   lstm_val_metrics["pr_auc"],
            "lstm_val/recall":   lstm_val_metrics["recall"],
            "lstm_val/fn":       lstm_val_metrics["fn"],
        }, step=epoch)

        print(f"epoch {epoch:>2}  train_loss={train_loss:.4f}  "
              f"ens_pr_auc={ens_val_metrics['pr_auc']:.4f}  "
              f"ens_recall={ens_val_metrics['recall']:.4f}  "
              f"ens_FN={ens_val_metrics['fn']}"
              f"  (lstm_pr_auc={lstm_val_metrics['pr_auc']:.4f})")

        # --- Checkpoint: best ensemble PR-AUC ---
        if ens_val_metrics["pr_auc"] > best_pr_auc_val:
            best_pr_auc_val   = ens_val_metrics["pr_auc"]
            best_pr_auc_state = copy.deepcopy(model.state_dict())
            best_pr_auc_epoch = epoch
            torch.save(best_pr_auc_state, pr_auc_ckpt)

        # --- Checkpoint: lowest ensemble FN ---
        if ens_val_metrics["fn"] < min_fn_val:
            min_fn_val        = ens_val_metrics["fn"]
            best_fn_state     = copy.deepcopy(model.state_dict())
            best_fn_epoch     = epoch
            best_fn_threshold = ens_val_metrics["threshold"]
            torch.save(best_fn_state, fn_ckpt)

    # -----------------------------------------------------------------------
    # 8. Final evaluation on both checkpoints
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Training done. Evaluating two best ensemble checkpoints.")
    print("=" * 60)

    if best_pr_auc_state is not None:
        _ensemble_final_eval_and_dump(
            model, best_pr_auc_state,
            val_loader, te_loader, y_val, y_te,
            xgb_val_score, xgb_test_score,
            args.xgb_weight, device,
            args.out_dir, args.run_name, tag="best_pr_auc",
            meta={"epoch": best_pr_auc_epoch,
                  "criterion": "max ensemble val PR-AUC",
                  "val_pr_auc_at_pick": best_pr_auc_val},
        )

    if best_fn_state is not None:
        ens_val_score_final, _, _ = _ensemble_final_eval_and_dump(
            model, best_fn_state,
            val_loader, te_loader, y_val, y_te,
            xgb_val_score, xgb_test_score,
            args.xgb_weight, device,
            args.out_dir, args.run_name, tag="best_fn",
            meta={"epoch": best_fn_epoch,
                  "criterion": "min ensemble val FN at F2-optimal threshold",
                  "val_fn_at_pick": int(min_fn_val),
                  "threshold_at_pick": float(best_fn_threshold) if best_fn_threshold else None},
        )
        # PR curve from the best-FN ensemble model
        probs_2col = np.stack([1.0 - ens_val_score_final, ens_val_score_final], axis=1)
        wandb.log({"best_fn/val/pr_curve": wandb.plot.pr_curve(
            y_val, probs_2col, labels=["no_fire", "fire"], classes_to_plot=[1]
        )})

    wandb.finish()


if __name__ == "__main__":
    main()
