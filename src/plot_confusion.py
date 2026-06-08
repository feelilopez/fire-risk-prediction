"""Render val + test confusion matrices from a `<run>-metrics.json` file.

Reads the JSON produced by `src.train_baseline`, draws a 1x2 figure (val
on the left, test on the right) with raw counts plus row-normalized
percentages, and saves a PNG next to the JSON.

Run:
    python -m src.plot_confusion runs/baseline-v1-metrics.json
    python -m src.plot_confusion runs/baseline-v1-metrics.json --out cm.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _cm_panel(ax, m: dict, title: str):
    """Draw one 2x2 confusion matrix panel with counts + row-% annotations."""
    # Build the 2x2 matrix in [actual, predicted] = [TN, FP; FN, TP] order.
    cm = np.array([[m["tn"], m["fp"]],
                   [m["fn"], m["tp"]]], dtype=int)
    # Row-normalize for color: each actual-class row sums to 1.
    # (Column-normalizing would mix the imbalanced denominators.)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / np.maximum(row_sums, 1)

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)

    # Annotate each cell with "count\n(row %)".
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_pct[i, j] * 100
            # White text on dark backgrounds, black on light.
            color = "white" if cm_pct[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", color=color, fontsize=12,
                    fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred: no fire", "pred: fire"])
    ax.set_yticklabels(["actual: no fire", "actual: fire"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")

    # Highlight the FN cell with a red border — it's the one we want low.
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-0.5, 0.5), 1, 1, fill=False,
                           edgecolor="red", linewidth=2.5))

    subtitle = (
        f"thr={m['threshold']:.3f}  "
        f"recall={m['recall']:.3f}  "
        f"prec={m['precision']:.3f}  "
        f"F2={m['f2']:.3f}\n"
        f"PR-AUC={m['pr_auc']:.3f}  "
        f"ROC-AUC={m['roc_auc']:.3f}  "
        f"miss_rate={m['fn_rate']:.3f}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path,
                        help="Path to <run_name>-metrics.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: same dir, same stem)")
    args = parser.parse_args()

    data = json.loads(args.metrics_json.read_text())
    val_m, test_m = data["val"], data["test"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _cm_panel(axes[0], val_m, "Validation")
    _cm_panel(axes[1], test_m, "Test")

    fig.suptitle(
        f"Confusion matrices — {args.metrics_json.stem}\n"
        f"red box = false negatives (missed fires)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    out = args.out or args.metrics_json.with_suffix(".cm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
