"""Evaluate a saved checkpoint (LSTM or XGBoost) at a chosen F-beta threshold.

Model type is auto-detected from the file extension:
  .pt   → FireLSTM (src.train_lstm)
  .json → XGBoost booster (src.train_baseline)

Usage
-----
# LSTM at F5
python -m src.eval_threshold \
    --checkpoint runs/my-run-best_fn.pt \
    --data-dir   data/lstm/clean \
    --beta 5

# XGBoost baseline at F5
python -m src.eval_threshold \
    --checkpoint runs/baseline-v1.json \
    --data-dir   data/baseline/clean \
    --beta 5

# Skip sweep, fix a threshold directly
python -m src.eval_threshold \
    --checkpoint runs/my-run-best_fn.pt \
    --data-dir   data/lstm/clean \
    --fixed-thr  0.15
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import evaluate, threshold_sweep
from src.train_lstm import FireLSTM, load_sequences, predict


# XGBoost was trained with sanitized names — must match at inference time.
_XGB_BAD = re.compile(r"[\[\]<]")

def _sanitize(names: list[str]) -> list[str]:
    return [_XGB_BAD.sub("_", n) for n in names]


# ---------------------------------------------------------------------------
# Model-specific loaders — each returns (val_scores, test_scores, y_val, y_te)
# ---------------------------------------------------------------------------

def _predict_lstm(checkpoint: Path, data_dir: Path,
                  hidden: int, n_layers: int, batch_size: int, device: str):
    meta         = joblib.load(data_dir / "meta.joblib")
    feature_cols = meta["feature_cols"]
    target       = meta["target"]

    X_val, y_val = load_sequences(data_dir, "val",  feature_cols, target)
    X_te,  y_te  = load_sequences(data_dir, "test", feature_cols, target)
    n_features   = X_val.shape[2]

    print(f"[eval] LSTM  val={X_val.shape}  test={X_te.shape}  features={n_features}")

    model = FireLSTM(n_features=n_features, hidden=hidden, n_layers=n_layers).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[eval] loaded weights from {checkpoint}")

    def _loader(X, y):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    return (
        predict(model, _loader(X_val, y_val), device),
        predict(model, _loader(X_te,  y_te),  device),
        y_val, y_te,
    )


def _predict_xgboost(checkpoint: Path, data_dir: Path):
    try:
        import xgboost as xgb
        import pandas as pd
    except ImportError:
        raise ImportError("xgboost and pandas must be installed for XGBoost evaluation.")

    meta         = joblib.load(data_dir / "meta.joblib")
    feature_cols = meta["feature_cols"]
    target       = meta["target"]
    san_cols     = _sanitize(feature_cols)

    val_df  = pd.read_parquet(data_dir / "val",  engine="pyarrow")
    test_df = pd.read_parquet(data_dir / "test", engine="pyarrow")

    # Fill any rare categories missing from val/test (mirrors train_baseline.py)
    for df in (val_df, test_df):
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0

    y_val = val_df[target].to_numpy(dtype=np.float32)
    y_te  = test_df[target].to_numpy(dtype=np.float32)

    print(f"[eval] XGBoost  val={val_df.shape}  test={test_df.shape}  features={len(feature_cols)}")

    booster = xgb.Booster()
    booster.load_model(str(checkpoint))
    print(f"[eval] loaded booster from {checkpoint}")

    dval  = xgb.DMatrix(val_df[feature_cols].to_numpy(dtype=np.float32),  feature_names=san_cols)
    dte   = xgb.DMatrix(test_df[feature_cols].to_numpy(dtype=np.float32), feature_names=san_cols)

    return booster.predict(dval), booster.predict(dte), y_val, y_te


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _cm_panel(ax, m: dict, title: str, beta: int):
    """2×2 confusion matrix with counts + row-normalised percentages."""
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=int)
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]*100:.1f}%)",
                    ha="center", va="center", color=color,
                    fontsize=12, fontweight="bold")

    # Red border around FN cell: actual=fire (row 1), pred=no-fire (col 0)
    ax.add_patch(Rectangle((-0.5, 0.5), 1, 1,
                            fill=False, edgecolor="red", linewidth=2.5))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred: no fire", "pred: fire"])
    ax.set_yticklabels(["actual: no fire", "actual: fire"])

    fbeta_val = m.get(f"f{beta}", float("nan"))
    ax.set_title(
        f"{title}\n"
        f"thr={m['threshold']:.3f}  recall={m['recall']:.3f}  "
        f"prec={m['precision']:.3f}  F{beta}={fbeta_val:.3f}\n"
        f"PR-AUC={m['pr_auc']:.3f}  miss_rate={m['fn_rate']:.3f}  "
        f"FN={m['fn']}  TP={m['tp']}",
        fontsize=10,
    )


def _plot_sweep(ax, sweep: dict, chosen_thr: float, beta: int):
    """Recall / precision / Fbeta vs threshold, with a marker at the chosen cut-off."""
    thrs = sweep["thresholds"]
    ax.plot(thrs, sweep["recall"],    label="Recall",    color="steelblue")
    ax.plot(thrs, sweep["precision"], label="Precision", color="darkorange")
    ax.plot(thrs, sweep["fbeta"],     label=f"F{beta}",  color="green")
    ax.axvline(chosen_thr, color="red", linestyle="--",
               label=f"chosen={chosen_thr:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"Val threshold sweep  (F{beta})")
    ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an LSTM or XGBoost checkpoint at a chosen F-beta threshold."
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help=".pt for LSTM, .json for XGBoost booster")
    parser.add_argument("--data-dir",   type=Path, required=True,
                        help="data/lstm/clean for LSTM, data/baseline/clean for XGBoost")
    parser.add_argument("--beta",       type=float, default=5.0,
                        help="Beta for threshold sweep (default 5; use 2 to match training)")
    parser.add_argument("--model-type", choices=["auto", "lstm", "xgboost"], default="auto",
                        help="Auto-detected from file extension if 'auto'")
    # LSTM architecture — must match the saved checkpoint
    parser.add_argument("--hidden",     type=int, default=128)
    parser.add_argument("--n-layers",   type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device",     type=str, default="cpu",
                        help="'cuda' or 'cpu'. Use cpu for local inspection of cluster checkpoints.")
    # Output
    parser.add_argument("--out-dir",   type=Path, default=None,
                        help="Directory to write PNG + JSON (default: same dir as checkpoint)")
    parser.add_argument("--fixed-thr", type=float, default=None,
                        help="Skip the sweep and use this threshold directly")
    args = parser.parse_args()

    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Auto-detect model type ---
    model_type = args.model_type
    if model_type == "auto":
        ext_map = {".pt": "lstm", ".json": "xgboost"}
        model_type = ext_map.get(args.checkpoint.suffix.lower())
        if model_type is None:
            raise ValueError(
                f"Cannot auto-detect model type from '{args.checkpoint.suffix}'. "
                f"Use --model-type lstm or --model-type xgboost."
            )
    print(f"[eval] model_type={model_type}  checkpoint={args.checkpoint.name}")

    # --- Load model and run inference ---
    if model_type == "lstm":
        # Fall back to cpu if cuda unavailable (common for local inspection)
        device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        if device != args.device:
            print(f"[eval] CUDA unavailable, falling back to cpu")
        val_scores, test_scores, y_val, y_te = _predict_lstm(
            args.checkpoint, args.data_dir,
            args.hidden, args.n_layers, args.batch_size, device,
        )
    else:
        val_scores, test_scores, y_val, y_te = _predict_xgboost(
            args.checkpoint, args.data_dir,
        )

    print(f"[eval] val   N={len(y_val):,}  fires={int(y_val.sum()):,} ({y_val.mean()*100:.1f}%)")
    print(f"[eval] test  N={len(y_te):,}   fires={int(y_te.sum()):,} ({y_te.mean()*100:.1f}%)")

    # --- Threshold selection ---
    sweep      = threshold_sweep(y_val, val_scores, beta=args.beta)
    beta_int   = int(args.beta)

    if args.fixed_thr is not None:
        chosen_thr = args.fixed_thr
        print(f"[eval] using fixed threshold: {chosen_thr:.4f}")
    else:
        chosen_thr = sweep["best_threshold"]
        print(f"[eval] F{beta_int}-optimal threshold on val: {chosen_thr:.4f}  "
              f"recall={sweep['best_recall']:.4f}  precision={sweep['best_precision']:.4f}")

    # --- Evaluate at chosen threshold ---
    val_m  = evaluate(y_val, val_scores,  beta=args.beta, fixed_threshold=chosen_thr)
    test_m = evaluate(y_te,  test_scores, beta=args.beta, fixed_threshold=chosen_thr)
    # Also compute F2 at the same threshold so you can compare with training-time metrics
    val_f2  = evaluate(y_val, val_scores,  beta=2.0, fixed_threshold=chosen_thr)
    test_f2 = evaluate(y_te,  test_scores, beta=2.0, fixed_threshold=chosen_thr)

    # --- Print summary table ---
    print()
    print("=" * 64)
    print(f"  {model_type.upper()}  |  threshold={chosen_thr:.4f}  (F{beta_int} on val)")
    print("=" * 64)
    print(f"{'metric':22s}  {'VAL':>10s}  {'TEST':>10s}")
    print("-" * 46)
    rows = [
        ("PR-AUC",         "pr_auc"),
        ("ROC-AUC",        "roc_auc"),
        ("Recall",         "recall"),
        ("Precision",      "precision"),
        (f"F{beta_int}",   f"f{beta_int}"),
        ("F2 (same thr)",  None),          # special: from val_f2 / test_f2
        ("TP",             "tp"),
        ("FN  ← minimize", "fn"),
        ("FP",             "fp"),
        ("TN",             "tn"),
        ("Miss rate (FN%)", "fn_rate"),
    ]
    for label, key in rows:
        if key is None:                    # F2 cross-check row
            v, t = val_f2["f2"], test_f2["f2"]
            print(f"{label:22s}  {v:>10.4f}  {t:>10.4f}")
        elif isinstance(val_m[key], int):
            print(f"{label:22s}  {val_m[key]:>10d}  {test_m[key]:>10d}")
        else:
            print(f"{label:22s}  {val_m[key]:>10.4f}  {test_m[key]:>10.4f}")
    print("=" * 64)

    # --- Save metrics JSON ---
    beta_tag  = f"f{beta_int}" if args.fixed_thr is None else f"thr{args.fixed_thr:.3f}"
    stem      = f"{args.checkpoint.stem}-{beta_tag}"
    json_path = out_dir / f"{stem}-metrics.json"
    json_path.write_text(json.dumps({
        "checkpoint":          str(args.checkpoint),
        "model_type":          model_type,
        "beta":                args.beta,
        "operating_threshold": chosen_thr,
        "val":                 val_m,
        "test":                test_m,
    }, indent=2))
    print(f"\n[eval] metrics → {json_path}")

    # --- Plot: sweep curve (top) + val/test confusion matrices (bottom) ---
    fig = plt.figure(figsize=(15, 10))

    ax_curve = fig.add_subplot(2, 1, 1)
    _plot_sweep(ax_curve, sweep, chosen_thr, beta_int)

    ax_val  = fig.add_subplot(2, 2, 3)
    ax_test = fig.add_subplot(2, 2, 4)
    _cm_panel(ax_val,  val_m,  "Validation", beta_int)
    _cm_panel(ax_test, test_m, "Test",       beta_int)

    fig.suptitle(
        f"{args.checkpoint.stem}  [{model_type}]  —  F{beta_int} threshold={chosen_thr:.3f}\n"
        f"red box = false negatives (missed fires)",
        fontsize=13,
    )
    fig.tight_layout()

    png_path = out_dir / f"{stem}-confusion.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[eval] plot    → {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
