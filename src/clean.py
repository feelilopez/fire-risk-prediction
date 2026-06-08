"""Clean sampled IberFire parquets for baseline (XGBoost) and LSTM training.

Reads `data/{baseline,lstm}/{train,val,test}` parquet shards, applies the
cleaning steps described in the project plan, and writes cleaned parquets to
`--out-dir` plus a scaler + encoder + feature list in `meta.joblib`.

Run:
    python -m src.clean --mode baseline --in-dir data/baseline --out-dir data/baseline/clean
    python -m src.clean --mode lstm     --in-dir data/lstm     --out-dir data/lstm/clean
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns dropped unconditionally per the cleaning spec.
DROP_ALWAYS = [
    "is_spain", "is_waterbody", "is_sea",  # constant after sampling filters
    "is_near_fire",                          # potential label leakage
    "label",                                 # duplicate of is_fire (target)
    "x_coordinate", "y_coordinate",          # redundant with x, y
    "x_index", "y_index",                    # redundant with x, y (grid indices)
    "FWI",                                   # dropped weather feature
]

# Identifier columns: written to output for analysis but not used as features.
# x, y are intentionally NOT here — geography is a real signal and we want
# the model to use it. They flow through to the numeric feature set and get
# z-scored along with everything else.
ID_COLS = [
    "sample_id", "time", "sample_type",
    "day_offset",  # present only in the LSTM mode
]

TARGET = "is_fire"
ONEHOT_COLS = ["AutonomousCommunities"]

# CLC version buckets: pick the latest year <= sample year.
CLC_YEARS_AVAILABLE = (2006, 2012, 2018)


def _clc_year_for(sample_year: np.ndarray) -> np.ndarray:
    """Return the CLC version year to use for each sample year.

    Rule: pick the latest CLC year <= the sample year.
        2008 -> 2006,  2014 -> 2012,  2020 -> 2018.

    How: `np.searchsorted(years, x, side="right")` returns the index where
    each x *would be inserted* into the sorted `years` array to keep it
    sorted. Subtracting 1 gives the index of the largest year <= x.
    Example with years=[2006, 2012, 2018] and sample=[2008, 2014, 2020]:
        searchsorted -> [1, 2, 3],  minus 1 -> [0, 1, 2],
        years[[0,1,2]] -> [2006, 2012, 2018]
    `np.clip` is defensive for any sample_year < 2006 (would give idx=-1).
    """
    years = np.array(CLC_YEARS_AVAILABLE)
    idx = np.searchsorted(years, sample_year, side="right") - 1
    idx = np.clip(idx, 0, len(years) - 1)
    return years[idx]


def _collapse_clc(df: pd.DataFrame) -> pd.DataFrame:
    """Replace `CLC_{year}_{suffix}` with a single `CLC_{suffix}` chosen by row year.

    Also drops the raw numeric-class columns (`CLC_{year}_{N}` where N is a
    digit): those are the 44 fine-grained Corine codes, already summed into
    the named aggregate proportions (urban_fabric, forest, scrub, ...) which
    we keep. No information loss; ~44 fewer columns per year.
    """
    pat = re.compile(r"^CLC_(\d{4})_(.+)$")
    digit_only = re.compile(r"^\d+$")

    # 1) Drop raw numeric-class columns up front.
    raw_class_cols = []
    for c in df.columns:
        m = pat.match(c)
        if m and digit_only.match(m.group(2)):
            raw_class_cols.append(c)
    if raw_class_cols:
        df = df.drop(columns=raw_class_cols)

    # 2) Group the remaining CLC_{year}_{name} columns by suffix.
    # We build a nested dict like:
    #   { "forest_proportion": {2006: "CLC_2006_forest_proportion",
    #                            2012: "CLC_2012_forest_proportion",
    #                            2018: "CLC_2018_forest_proportion"},
    #     "urban_fabric_proportion": {...}, ... }
    #
    # `dict.setdefault(k, default)` returns dict[k] if k exists, otherwise
    # inserts default and returns it. So `d.setdefault(k, {})[k2] = v` is
    # the idiomatic one-liner for "make sure d[k] is a dict, then set d[k][k2]".
    suffix_to_yearcol: dict[str, dict[int, str]] = {}
    for c in df.columns:
        m = pat.match(c)
        if m:
            year, suffix = int(m.group(1)), m.group(2)
            suffix_to_yearcol.setdefault(suffix, {})[year] = c
    if not suffix_to_yearcol:
        return df

    # Per-row: which year should we pull from?
    sample_year = pd.to_datetime(df["time"]).dt.year.to_numpy()
    chosen_year = _clc_year_for(sample_year)

    # Build the collapsed columns in a temporary dict, then attach them in
    # one shot at the end. Building columns one-by-one with `df[c] = ...`
    # repeatedly fragments pandas internals and is much slower.
    new_cols = {}
    drop_cols = []
    for suffix, year_to_col in suffix_to_yearcol.items():
        years_avail = np.array(sorted(year_to_col.keys()))
        # In practice years_avail == CLC_YEARS_AVAILABLE, but be defensive
        # in case some suffix is missing one of the years.
        effective = years_avail[np.searchsorted(years_avail, chosen_year, side="right") - 1]
        effective = np.clip(effective, years_avail.min(), years_avail.max())

        # Build the collapsed array row-by-row using boolean masks:
        # for each available year, find which rows want that year, and copy
        # the values from that year's column into the matching positions.
        out = np.empty(len(df), dtype=np.float32)
        out[:] = np.nan  # safe default; gets overwritten in the loop below
        for yr, col in year_to_col.items():
            mask = effective == yr     # boolean array of length len(df)
            if mask.any():
                # df.loc[mask, col] picks rows where mask=True; .to_numpy
                # converts to a plain array we can assign into `out[mask]`.
                out[mask] = df.loc[mask, col].to_numpy(dtype=np.float32)
            drop_cols.append(col)
        new_cols[f"CLC_{suffix}"] = out

    df = df.drop(columns=drop_cols)
    for k, v in new_cols.items():
        df[k] = v
    return df


def _collapse_popdens(df: pd.DataFrame) -> pd.DataFrame:
    """Replace `popdens_{year}` with a single `popdens` chosen by nearest year."""
    pat = re.compile(r"^popdens_(\d{4})$")
    year_to_col: dict[int, str] = {}
    for c in df.columns:
        m = pat.match(c)
        if m:
            year_to_col[int(m.group(1))] = c
    if not year_to_col:
        return df

    years_avail = np.array(sorted(year_to_col.keys()))
    sample_year = pd.to_datetime(df["time"]).dt.year.to_numpy()

    # Nearest available year for each sample, vectorized via broadcasting.
    # `years_avail` has shape (Y,), `sample_year` has shape (N,).
    #   years_avail[None, :]   becomes shape (1, Y)  -- "add a row axis"
    #   sample_year[:, None]   becomes shape (N, 1)  -- "add a column axis"
    # Subtracting them broadcasts to shape (N, Y): for each sample (row),
    # the distance to every available year (col). argmin(axis=1) picks the
    # closest year per row.
    diff = np.abs(years_avail[None, :] - sample_year[:, None])
    idx = diff.argmin(axis=1)
    chosen_year = years_avail[idx]

    out = np.empty(len(df), dtype=np.float32)
    out[:] = np.nan
    for yr, col in year_to_col.items():
        mask = chosen_year == yr
        if mask.any():
            out[mask] = df.loc[mask, col].to_numpy(dtype=np.float32)

    df = df.drop(columns=list(year_to_col.values()))
    df["popdens"] = out
    return df


def _ffill_window(df: pd.DataFrame) -> pd.DataFrame:
    """LSTM mode: forward-fill nulls within each 7-day (sample_id) window.

    Sort by day_offset descending so the oldest day (offset=6) comes first;
    ffill then bfill propagates observations across the window in real time
    order. Columns where all 7 days are null remain null and get median-
    imputed downstream from the train medians.

    No-op when `day_offset` is absent (baseline mode).
    """
    if "day_offset" not in df.columns or "sample_id" not in df.columns:
        return df

    # Pick numeric columns only; ffill/bfill on strings or datetimes would
    # either be a no-op or error. `select_dtypes(include="number")` returns
    # all numeric dtypes (int*, float*, uint*).
    skip = {TARGET, "sample_id", "day_offset"}
    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in skip
    ]

    # Sort so that within each sample_id group, rows go from oldest day
    # (day_offset=6, i.e. 6 days before the sample date) to newest
    # (day_offset=0, the sample day). Ascending=[True, False] = sample_id
    # ascending, day_offset descending.
    df = df.sort_values(["sample_id", "day_offset"], ascending=[True, False])

    # Use native GroupBy methods ffill and bfill (much faster than Python lambda transform).
    # Since they return DataFrames with the original index, direct assignment works.
    g = df.groupby("sample_id", sort=False)[num_cols]
    df[num_cols] = g.ffill()
    df[num_cols] = g.bfill()
    return df


def _add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    """Derive month from `time` and encode as (sin, cos)."""
    m = pd.to_datetime(df["time"]).dt.month.to_numpy().astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * m / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * m / 12).astype(np.float32)
    return df


def _drop_known(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in DROP_ALWAYS if c in df.columns]
    return df.drop(columns=drop)


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (ids_present, categoricals_present, numeric_features).

    Splits the columns into three buckets so downstream code knows what to
    do with each:
      - ids: kept in the output but excluded from scaling/encoding.
      - cats: passed through OneHotEncoder.
      - num: passed through StandardScaler.
    """
    # List comprehensions: `[expr for x in iter if cond]` builds a new list
    # of `expr` for each `x` in `iter` where `cond` is True. Equivalent to
    # a small for-loop with an append, but ~2x faster and more readable.
    ids = [c for c in ID_COLS if c in df.columns]
    cats = [c for c in ONEHOT_COLS if c in df.columns]

    # Set union with `|`. Sets give O(1) `in` checks (vs O(n) for lists),
    # so this is the fast way to ask "is this column accounted for?".
    excluded = set(ids) | set(cats) | {TARGET}
    num = [c for c in df.columns if c not in excluded]

    # Defensive: anything still in `num` that isn't numeric (e.g. a stray
    # string column the sampling notebook adds in the future) would crash
    # the StandardScaler with "could not convert string to float".
    num = [c for c in num if pd.api.types.is_numeric_dtype(df[c])]
    return ids, cats, num


def load_split(in_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(in_dir / split, engine="pyarrow")


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_known(df)
    df = _add_cyclical_month(df)
    df = _collapse_clc(df)
    df = _collapse_popdens(df)
    # LSTM mode only: forward-fill nulls across the 7-day window per sample.
    # Whatever the window doesn't cover falls back to train-median imputation
    # in fit_transform_train / transform below.
    df = _ffill_window(df)
    return df


def fit_transform_train(df: pd.DataFrame):
    """Fit imputer/scaler/encoder on TRAIN and apply them to TRAIN.

    Returns (cleaned_df, ohe, scaler, feature_cols). The fitted objects are
    then handed to `transform()` for val/test — that's how we prevent
    information leaking from val/test back into the train statistics.
    """
    df = _preprocess(df)
    ids, cats, num = _split_columns(df)

    # 1) Impute numerics with the train medians.
    # `df[num].median(numeric_only=True)` returns a Series of per-column
    # medians. `fillna(series)` then replaces NaN in each column with that
    # column's median (pandas aligns by column name).
    medians = df[num].median(numeric_only=True)
    df[num] = df[num].fillna(medians)

    # 2) Z-score numerics: subtract mean, divide by std, fit on train.
    # `.astype(np.float32)` halves the memory vs the default float64.
    scaler = StandardScaler()
    df[num] = scaler.fit_transform(df[num]).astype(np.float32)

    # 3) One-hot encode categoricals (AutonomousCommunities).
    # `handle_unknown="ignore"` means if val/test contains a category that
    # never appeared in train, it produces all-zero columns instead of
    # crashing. `sparse_output=False` returns a dense numpy array (we have
    # ~19 categories, sparse storage isn't worth the complexity).
    if cats:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        ohe_arr = ohe.fit_transform(df[cats].astype(str))
        ohe_cols = list(ohe.get_feature_names_out(cats))
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
    else:
        ohe = None
        ohe_df = pd.DataFrame(index=df.index)
        ohe_cols = []

    # 4) Assemble output: IDs + target + scaled numerics + OHE columns.
    # `df[ids + [TARGET] + num]` uses list concatenation to pick columns in
    # a specific order. `pd.concat(..., axis=1)` glues frames side-by-side
    # (axis=0 would stack rows). Both frames share the same index (sample
    # row numbers) so concat aligns them correctly.
    feature_cols = num + ohe_cols
    out = pd.concat([df[ids + [TARGET] + num], ohe_df], axis=1)

    # 5) Pragmatic hack: attach the medians + column list as attributes on
    # the scaler so transform() can be called with just (df, ohe, scaler)
    # and re-apply the exact same pipeline. The cleaner alternative is a
    # sklearn Pipeline + ColumnTransformer, but that's more boilerplate for
    # the same outcome. type: ignore tells the type checker we know we're
    # adding non-standard attributes.
    scaler.medians_ = medians                # type: ignore[attr-defined]
    scaler.numeric_cols_ = num               # type: ignore[attr-defined]
    return out, ohe, scaler, feature_cols


def transform(df: pd.DataFrame, ohe, scaler) -> pd.DataFrame:
    df = _preprocess(df)
    ids, cats, _ = _split_columns(df)
    num: list[str] = scaler.numeric_cols_           # type: ignore[attr-defined]
    medians: pd.Series = scaler.medians_            # type: ignore[attr-defined]

    for c in num:
        if c not in df.columns:
            df[c] = medians[c]
    df[num] = df[num].fillna(medians)
    df[num] = scaler.transform(df[num]).astype(np.float32)

    if ohe is not None and cats:
        ohe_arr = ohe.transform(df[cats].astype(str))
        ohe_cols = list(ohe.get_feature_names_out(cats))
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
    else:
        ohe_df = pd.DataFrame(index=df.index)

    return pd.concat([df[ids + [TARGET] + num], ohe_df], axis=1)


def write_split(df: pd.DataFrame, out_dir: Path, split: str):
    out_path = out_dir / split
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path / "part-00001.parquet", engine="pyarrow", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "lstm"], required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[clean] loading train from {args.in_dir}/train")
    train = load_split(args.in_dir, "train")
    print(f"[clean] train shape: {train.shape}")

    print("[clean] fitting transforms on train")
    train_clean, ohe, scaler, feature_cols = fit_transform_train(train)
    print(f"[clean] train_clean shape: {train_clean.shape}, features: {len(feature_cols)}")
    write_split(train_clean, args.out_dir, "train")
    del train, train_clean

    for split in ("val", "test"):
        print(f"[clean] loading {split}")
        df = load_split(args.in_dir, split)
        df_clean = transform(df, ohe, scaler)
        print(f"[clean] {split}_clean shape: {df_clean.shape}")
        write_split(df_clean, args.out_dir, split)
        del df, df_clean

    meta = {
        "ohe": ohe,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target": TARGET,
        "id_cols": ID_COLS,
        "mode": args.mode,
    }
    joblib.dump(meta, args.out_dir / "meta.joblib")
    print(f"[clean] wrote meta to {args.out_dir/'meta.joblib'}")


if __name__ == "__main__":
    main()
