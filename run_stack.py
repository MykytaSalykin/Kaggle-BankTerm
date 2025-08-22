#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime

PROJ = Path.cwd()
DATA = next(
    (
        p
        for p in [PROJ / "data", PROJ.parent / "data", PROJ.parent.parent / "data"]
        if (p / "train.csv").exists()
    ),
    None,
)
assert DATA is not None, "data/train.csv not found"

OUT = PROJ / "notebooks" / "outputs"
CACHE = OUT / "cache"
SUB = OUT / "submissions"
SUB.mkdir(parents=True, exist_ok=True)


def load_cache(name):
    o = np.load(CACHE / f"{name}_oof.npy")
    t = np.load(CACHE / f"{name}_test.npy")
    return o, t


# Expect cached OOF/test arrays produced by 02_baselines.ipynb
NAMES = ["lgbF_s42", "lgbF_s7", "lgbF_spw", "catF_cpu", "xgbF_ohe"]
loaded = [load_cache(n) for n in NAMES]
oofs = [o for o, _ in loaded]
tests = [t for _, t in loaded]

y = pd.read_csv(DATA / "train.csv")["y"].values
Z_tr = np.vstack(oofs).T
Z_te = np.vstack(tests).T

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta = np.zeros(len(y))
pred_meta = np.zeros(len(Z_te))
for tr, va in skf.split(Z_tr, y):
    m = LogisticRegression(max_iter=1000)
    m.fit(Z_tr[tr], y[tr])
    oof_meta[va] = m.predict_proba(Z_tr[va])[:, 1]
    pred_meta += m.predict_proba(Z_te)[:, 1] / skf.n_splits

print("Stack OOF AUC:", roc_auc_score(y, oof_meta))
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_path = SUB / f"final_stack_{ts}.csv"
pd.DataFrame({"id": pd.read_csv(DATA / "test.csv")["id"], "y": pred_meta}).to_csv(
    sub_path, index=False
)
print("Saved:", sub_path)
