# advanced/features.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_NUM = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
BASE_CAT = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


def _month_to_num(m: str) -> int:
    d = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return d.get(str(m).lower(), 0)


def _curated(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # safe clip for duration to limit tail
    dur_clip = np.minimum(out["duration"], np.quantile(out["duration"], 0.99))
    out["duration_clip_99"] = dur_clip
    out["duration_log1p"] = np.log1p(dur_clip)
    out["duration_per_call"] = dur_clip / (out["campaign"] + 1.0)

    out["pdays_was_contacted"] = (out["pdays"] != -1).astype(int)
    out["pdays_pos_log"] = np.log1p(
        out["pdays"].where(out["pdays"] != -1, np.nan)
    ).fillna(0.0)
    out["previous_gt0"] = (out["previous"] > 0).astype(int)

    # month cyclic
    out["month_num"] = out["month"].map(_month_to_num).astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12.0)
    # day cyclic
    out["day_sin"] = np.sin(2 * np.pi * out["day"] / 31.0)
    out["day_cos"] = np.cos(2 * np.pi * out["day"] / 31.0)

    # contact × duration
    out["contact_cellular"] = (out["contact"] == "cellular").astype(int)
    out["dur_x_cell"] = out["duration_clip_99"] * out["contact_cellular"]

    # simple bins that generalize well
    out["duration_q10"] = pd.qcut(dur_clip, q=10, labels=False, duplicates="drop")
    out["campaign_q5"] = pd.qcut(out["campaign"], q=5, labels=False, duplicates="drop")

    # robust string combos (kept as category for CatBoost)
    out["job_education"] = out["job"].astype(str) + "_" + out["education"].astype(str)
    out["contact_month_day"] = (
        out["contact"].astype(str)
        + "_"
        + out["month"].astype(str)
        + "_"
        + out["day"].astype(str)
    )
    return out


def _freq_encode(full: pd.Series, ref: pd.Series) -> pd.Series:
    """Frequency encoding that works even if `full` is categorical."""
    freq = ref.value_counts(normalize=True)
    # Cast to object to avoid Categorical fillna issues, then to float
    s = full.astype(object).map(freq)
    return s.fillna(0.0).astype("float32")


def _oof_target_encode(
    col_full: pd.Series, y_train: np.ndarray, n_train: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """OOF target-encoding on train-part; test-part via full mapping."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(n_train, dtype=np.float32)
    for tr, va in skf.split(np.zeros(n_train), y_train):
        m = pd.DataFrame({"c": col_full.iloc[:n_train].values, "y": y_train})
        means = m.iloc[tr].groupby("c")["y"].mean()
        oof[va] = (
            m.iloc[va]["c"]
            .map(means)
            .fillna(y_train[tr].mean())
            .astype(np.float32)
            .values
        )
    means_full = (
        pd.DataFrame({"c": col_full.iloc[:n_train].values, "y": y_train})
        .groupby("c")["y"]
        .mean()
    )
    test_enc = (
        col_full.iloc[n_train:]
        .map(means_full)
        .fillna(y_train.mean())
        .astype(np.float32)
        .values
    )
    return oof, test_enc


def build_features(
    X_comb: pd.DataFrame, X_test: pd.DataFrame, y: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    X_comb: train (+ optional bank-full) features without id/y.
    X_test: test features without id.
    y: target for X_comb.
    Returns: (X_feat_comb, X_feat_test)
    """
    n_train = len(X_comb)
    df_all = pd.concat(
        [X_comb.reset_index(drop=True), X_test.reset_index(drop=True)],
        axis=0,
        ignore_index=True,
    )

    # curated core
    df_all = _curated(df_all)

    # ensure categorical dtype for base cats and combos
    cat_like = list(set(BASE_CAT + ["job_education", "contact_month_day"]))
    for c in cat_like:
        if c in df_all.columns:
            df_all[c] = df_all[c].astype("category")

    # frequency encoding for base cats (computed on train-part)
    for c in BASE_CAT:
        if c in df_all.columns:
            df_all[f"{c}_freq"] = _freq_encode(
                df_all.loc[: n_train - 1, c], df_all[c]
            ).astype(np.float32)

    # OOF target-encoding on selected columns
    te_cols = [
        c
        for c in ["job", "month", "poutcome", "job_education", "contact_month_day"]
        if c in df_all.columns
    ]
    for c in te_cols:
        oof_enc, test_enc = _oof_target_encode(
            df_all[c].astype(str), y, n_train, seed=42
        )
        df_all[f"{c}_te"] = np.r_[oof_enc, test_enc]

    # bin-wise target rates (duration, campaign) — light target enc
    for bcol, outname in [("duration_q10", "durq10_te"), ("campaign_q5", "campq5_te")]:
        oof_enc, test_enc = _oof_target_encode(
            df_all[bcol].astype(str), y, n_train, seed=123
        )
        df_all[outname] = np.r_[oof_enc, test_enc]

    # split back
    X_feat = df_all.iloc[:n_train].copy()
    X_test_feat = df_all.iloc[n_train:].copy()

    # keep original cats as categorical for CatBoost; others numeric
    for c in cat_like:
        if c in X_feat.columns:
            X_feat[c] = X_feat[c].astype("category")
            X_test_feat[c] = X_test_feat[c].astype("category")

    return X_feat, X_test_feat
