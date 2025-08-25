# advanced/common.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from advanced.features import BASE_CAT, BASE_NUM, build_features


# ------------ paths / io ------------


def resolve_data_dir() -> Path:
    """
    Find the folder that contains train.csv and test.csv.
    Search current repo and a couple of parent locations.
    """
    proj = Path.cwd()
    candidates = [
        proj / "data",
        proj / "notebooks" / "data",
        proj.parent / "data",
        proj.parent / "notebooks" / "data",
        proj.parent.parent / "data",
    ]
    for p in candidates:
        if (p / "train.csv").exists() and (p / "test.csv").exists():
            return p
    # last resort: rglob
    for base in [proj] + list(proj.parents)[:3]:
        for f in base.rglob("train.csv"):
            if (f.parent / "test.csv").exists():
                return f.parent
    raise FileNotFoundError("Could not find data folder with train.csv and test.csv")


def find_orig_bank_full() -> Path | None:
    """
    Try to locate original UCI/Kaggle 'bank-full.csv' (semicolon separated).
    Returns a Path or None if not found.
    """
    proj = Path.cwd()
    names = ["bank-full.csv", "bank.csv", "bank-full/bank-full.csv"]
    roots = [
        proj / "data",
        proj / "notebooks" / "data",
        proj.parent / "data",
        proj.parent / "notebooks" / "data",
        proj.parent.parent / "data",
    ]
    for r in roots:
        for nm in names:
            p = r / nm
            if p.exists():
                return p
    for base in [proj] + list(proj.parents)[:3]:
        for f in base.rglob("bank-full.csv"):
            return f
    return None


def ensure_outputs():
    """
    Create notebooks/outputs/{cache,submissions} and return (OUT, CACHE, SUB) paths.
    """
    proj = Path.cwd()
    out = proj / "notebooks" / "outputs"
    cache = out / "cache"
    sub = out / "submissions"
    for p in (out, cache, sub):
        p.mkdir(parents=True, exist_ok=True)
    return out, cache, sub


def save_cache(cache_dir: Path, name: str, oof: np.ndarray, pred: np.ndarray) -> None:
    np.save(cache_dir / f"{name}_oof.npy", oof)
    np.save(cache_dir / f"{name}_test.npy", pred)


def load_cache(cache_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    o = np.load(cache_dir / f"{name}_oof.npy")
    t = np.load(cache_dir / f"{name}_test.npy")
    return o, t


# ------------ data prep ------------


def _normalize_bank_full_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align original bank-full to competition schema (lowercase + exact column order).
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    needed = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"bank-full.csv missing columns: {missing}")
    return df[needed].copy()


def prepare_frames(
    use_orig: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, bool]:
    """
    Load train/test; optionally augment with original bank-full (train+orig concatenation).
    Build features with advanced.features.build_features.
    Returns (X_feat, y, X_test_feat, used_orig_flag).
    """
    data_dir = resolve_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    y_train = train["y"].astype(int).values
    X_base = train.drop(columns=["id", "y"]).copy()
    X_test_base = test.drop(columns=["id"]).copy()

    used_orig = False
    if use_orig:
        p = find_orig_bank_full()
        if p is not None:
            sep = ";" if p.suffix.lower() == ".csv" else None
            orig = pd.read_csv(p, sep=sep)
            if "y" in orig.columns:
                # map "yes"/"no" -> 1/0 if needed
                if orig["y"].dtype == object:
                    orig["y"] = orig["y"].map({"yes": 1, "no": 0})
            orig = _normalize_bank_full_columns(orig)
            X_orig = orig.drop(columns=["y"]).copy()
            y_orig = orig["y"].astype(int).values

            # concat train-like features with orig
            X_comb = pd.concat([X_base, X_orig], ignore_index=True)
            y_comb = np.concatenate([y_train, y_orig])
            used_orig = True
        else:
            X_comb, y_comb = X_base, y_train
    else:
        X_comb, y_comb = X_base, y_train

    # feature engineering
    X_feat, X_test_feat = build_features(X_comb, X_test_base, y=y_comb)

    return X_feat, y_comb, X_test_feat, used_orig
