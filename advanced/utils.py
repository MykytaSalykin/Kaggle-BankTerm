# advanced/utils.py
from pathlib import Path
import os
import numpy as np

PROJ = Path.cwd()


def resolve_data_dir():
    env = os.getenv("DATA_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "train.csv").exists() and (p / "test.csv").exists():
            return p
    for p in [PROJ / "data", PROJ.parent / "data", PROJ.parent.parent / "data"]:
        if (p / "train.csv").exists() and (p / "test.csv").exists():
            return p
    return None


def find_orig_bank_full():
    cands = [
        PROJ / "data" / "bank-full.csv",
        PROJ / "data" / "bank" / "bank-full.csv",
        PROJ / "data" / "external" / "bank-full.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def ensure_outputs():
    out = PROJ / "notebooks" / "outputs"
    cache = out / "cache"
    sub = out / "submissions"
    for d in [out, cache, sub]:
        d.mkdir(parents=True, exist_ok=True)
    return out, cache, sub


def save_cache(cache_dir: Path, name: str, oof: np.ndarray, test: np.ndarray):
    np.save(cache_dir / f"{name}_oof.npy", oof)
    np.save(cache_dir / f"{name}_test.npy", test)


def load_cache(cache_dir: Path, name: str):
    o = np.load(cache_dir / f"{name}_oof.npy")
    t = np.load(cache_dir / f"{name}_test.npy")
    return o, t


def has_cache(cache_dir: Path, name: str) -> bool:
    return (cache_dir / f"{name}_oof.npy").exists() and (
        cache_dir / f"{name}_test.npy"
    ).exists()
