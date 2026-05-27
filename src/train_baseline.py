"""XGBoost baseline trainer for wildfire risk prediction.

Reads cleaned parquets from `data/baseline/clean/{train,val,test}` plus the
meta.joblib produced by `src.clean`, trains an XGBoost classifier on H100
GPU, logs to wandb (offline-friendly), and reports recall-focused metrics.

CALIBRATION NOTE: train is 1:3 (pos:neg) by design (sampling) and we apply
`scale_pos_weight` on top of that, so the model's raw predicted probabilities
are NOT calibrated to real-world fire prevalence (~0.0024%). We pick the
operating threshold on val (1:10 ratio) and apply it to test (also 1:10), so
the val/test comparison is apples-to-apples. For operational deployment the
threshold (or probabilities) would need recalibration against the true prior.

Run:
    python -m src.train_baseline --data-dir data/baseline/clean --run-name baseline-v1
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb

from src.metrics import evaluate, threshold_sweep


# XGBoost rejects feature names containing `[`, `]`, `<`. We never use those
# in our cleaned schema, but sanitize defensively in case a future encoder
# (e.g. interaction features) introduces them.
_XGB_BAD = re.compile(r"[\[\]<]")


def _sanitize_feature_names(names: list[str]) -> list[str]:
    return [_XGB_BAD.sub("_", n) for n in names]


def load_xy(data_dir: Path, split: str, feature_cols: list[str], target: str):
    df = pd.read_parquet(data_dir / split, engine="pyarrow")
    # Some feature cols may be missing in val/test (rare cats); fill with 0.
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.int32)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-pos-weight", type=float, default=None,
                        help="If None, computed from train class balance.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = joblib.load(args.data_dir / "meta.joblib")
    feature_cols = meta["feature_cols"]
    target = meta["target"]

    X_tr, y_tr = load_xy(args.data_dir, "train", feature_cols, target)
    X_val, y_val = load_xy(args.data_dir, "val", feature_cols, target)
    X_te, y_te = load_xy(args.data_dir, "test", feature_cols, target)

    if args.scale_pos_weight is None:
        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        spw = n_neg / max(n_pos, 1)
    else:
        spw = args.scale_pos_weight

    config = vars(args).copy()
    config.update({
        "n_features": len(feature_cols),
        "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
        "train_pos_rate": float(y_tr.mean()),
        "val_pos_rate": float(y_val.mean()),
        "test_pos_rate": float(y_te.mean()),
        "scale_pos_weight": float(spw),
    })
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "fire-risk"),
        name=args.run_name,
        config=config,
        dir=str(args.out_dir),
    )

    # NOTE: XGBoost uses the LAST metric in `eval_metric` for early stopping.
    # For our recall-focused, imbalanced setting `aucpr` (area under the
    # precision-recall curve) is the right stopping signal — logloss is
    # included for visibility only.
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "tree_method": "hist",
        "device": args.device,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": spw,
        "seed": args.seed,
    }

    safe_feature_names = _sanitize_feature_names(feature_cols)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=safe_feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=safe_feature_names)
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=safe_feature_names)

    evals_result: dict = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50,
    )

    # Log per-iteration curves to wandb.
    # `evals_result` is shaped like:
    #   {"train": {"logloss": [...], "aucpr": [...]},
    #    "val":   {"logloss": [...], "aucpr": [...]}}
    # We zip them together into one wandb.log() call per iteration so wandb
    # treats them as the same step and they line up on shared x-axis charts.
    n_iters = len(next(iter(next(iter(evals_result.values())).values())))
    for it in range(n_iters):
        row = {}
        for split, mdict in evals_result.items():
            for metric_name, values in mdict.items():
                row[f"{split}/{metric_name}"] = values[it]
        wandb.log(row, step=it)

    val_score = booster.predict(dval)
    test_score = booster.predict(dtest)

    # Pick operating threshold on val using F2 (recall-weighted), apply to test.
    val_sweep = threshold_sweep(y_val, val_score, beta=2.0)
    op_threshold = val_sweep["best_threshold"]

    val_metrics = evaluate(y_val, val_score, beta=2.0, fixed_threshold=op_threshold)
    test_metrics = evaluate(y_te, test_score, beta=2.0, fixed_threshold=op_threshold)

    # Flatten val/test metric dicts into a single wandb.log call.
    # The `**` operator unpacks a dict into a function call. `{**a, **b}`
    # merges two dicts (b's keys win on collision; here there are none).
    wandb.log({**{f"val/{k}": v for k, v in val_metrics.items()},
               **{f"test/{k}": v for k, v in test_metrics.items()}})

    # PR curve on val (our minimize-FN north star): wandb.plot.pr_curve takes
    # ground truth + class-probability matrix (N, n_classes). For binary we
    # stack [1-p, p] so it has columns for class 0 and class 1.
    val_probs_2col = np.stack([1.0 - val_score, val_score], axis=1)
    wandb.log({
        "val/pr_curve": wandb.plot.pr_curve(
            y_val, val_probs_2col, labels=["no_fire", "fire"], classes_to_plot=[1]
        ),
    })

    # Feature importance, logged as a wandb Table for sortable display.
    # `importance_type="gain"` = total reduction in loss this feature
    # contributed across all splits it was used for. Better signal than
    # "weight" (just the count of splits).
    fi = booster.get_score(importance_type="gain")
    fi_sorted = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
    fi_table = wandb.Table(columns=["feature", "gain"],
                           data=[[k, float(v)] for k, v in fi_sorted[:50]])
    wandb.log({"feature_importance": fi_table})

    # Persist artifacts.
    booster.save_model(args.out_dir / f"{args.run_name}.json")
    with open(args.out_dir / f"{args.run_name}-metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics,
                   "operating_threshold": op_threshold}, f, indent=2)

    print("val:", val_metrics)
    print("test:", test_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
