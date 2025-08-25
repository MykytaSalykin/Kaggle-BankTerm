# advanced/stack_adv.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PROJ = Path.cwd()
OUT = PROJ / "notebooks" / "outputs"
CACHE = OUT / "cache"
SUB = OUT / "submissions"
SUB.mkdir(parents=True, exist_ok=True)


def _load_csv_cols():
    candidates = [PROJ / "data", PROJ.parent / "data", PROJ.parent.parent / "data"]
    data_dir = next(
        (
            p
            for p in candidates
            if (p / "train.csv").exists() and (p / "test.csv").exists()
        ),
        None,
    )
    if data_dir is None:
        raise FileNotFoundError("data/train.csv or data/test.csv not found")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["y"].values.astype(int)
    ids = test["id"].values
    return y, ids


def _try_load_pair(name: str):
    oof_path = CACHE / f"{name}_oof.npy"
    test_path = CACHE / f"{name}_test.npy"
    if oof_path.exists() and test_path.exists():
        return np.load(oof_path), np.load(test_path)
    return None


def _gather_sets():
    candidates = [
        "adv_lgb_gbdt",
        "adv_cat",
        "adv_xgb_hist",
        "lgbF_s42",
        "lgbF_s7",
        "lgbF_spw",
        "catF_cpu",
        "xgbF_ohe",
    ]
    loaded = []
    for n in candidates:
        pair = _try_load_pair(n)
        if pair is not None:
            loaded.append((n, *pair))
    if len(loaded) < 2:
        raise RuntimeError("Not enough caches found to ensemble.")
    return loaded


def _rank_normalize(a: np.ndarray) -> np.ndarray:
    # strictly monotonic rank to [0,1]
    r = a.argsort().argsort().astype(float)
    r /= len(a) - 1
    return r


def _cv_score(y, pred, splits):
    oof = np.zeros_like(y, dtype=float)
    for tr, va in splits:
        # no fitting, just copy fold subset
        oof[va] = pred[va]
    return roc_auc_score(y, oof)


def _lr_stack(y, Z_tr, Z_te, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    pred = np.zeros(len(Z_te), dtype=float)
    for tr, va in skf.split(Z_tr, y):
        m = LogisticRegression(max_iter=2000)
        m.fit(Z_tr[tr], y[tr])
        oof[va] = m.predict_proba(Z_tr[va])[:, 1]
        pred += m.predict_proba(Z_te)[:, 1] / skf.n_splits
    return oof, pred


def _grid_three(y, preds, labels):
    # simple weight grid for top-3 models if present
    # returns best name, oof, test
    name_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    keys = ["adv_lgb_gbdt", "adv_cat", "adv_xgb_hist"]
    if not all(k in name_to_idx for k in keys):
        return None
    iL, iC, iX = [name_to_idx[k] for k in keys]
    pL, pC, pX = preds[iL], preds[iC], preds[iX]

    best = (-1.0, None, None, None)
    for wL in np.linspace(0.2, 0.7, 26):  # 0.2 .. 0.7 step 0.02
        for wC in np.linspace(0.1, 0.6, 26):
            wX = 1.0 - wL - wC
            if wX < 0.05:
                continue
            oof = wL * pL[0] + wC * pC[0] + wX * pX[0]
            score = roc_auc_score(y, oof)
            if score > best[0]:
                best = (score, (wL, wC, wX), wL * pL[1] + wC * pC[1] + wX * pX[1], oof)
    if best[1] is None:
        return None
    score, weights, test_pred, oof = best
    tag = f"grid3_L{weights[0]:.2f}_C{weights[1]:.2f}_X{weights[2]:.2f}"
    return (tag, score, oof, test_pred)


def main():
    y, test_ids = _load_csv_cols()
    loaded = _gather_sets()
    names = [n for (n, _, _) in loaded]
    oofs = [o for (_, o, _) in loaded]
    tests = [t for (_, _, t) in loaded]

    # report individual CVs
    print("Found caches:", names)
    for n, o in zip(names, oofs):
        print(f"OOF[{n}]: {roc_auc_score(y, o):.6f}")

    Z_tr = np.vstack(oofs).T
    Z_te = np.vstack(tests).T

    # 1) LR stack
    oof_lr, pred_lr = _lr_stack(y, Z_tr, Z_te, seed=42)
    score_lr = roc_auc_score(y, oof_lr)
    print(f"LR stack OOF: {score_lr:.6f}")

    # 2) Rank average (all)
    oofs_rank = np.vstack([_rank_normalize(o) for o in oofs])
    tests_rank = np.vstack([_rank_normalize(t) for t in tests])

    oof_rank_avg = oofs_rank.mean(axis=0)
    test_rank_avg = tests_rank.mean(axis=0)
    score_ra = roc_auc_score(y, oof_rank_avg)
    print(f"Rank-avg(all) OOF: {score_ra:.6f}")

    # 3) Small 3-model weight grid if adv_* all present
    grid_res = _grid_three(y, list(zip(oofs, tests)), names)
    if grid_res is not None:
        tag3, score3, oof3, test3 = grid_res
        print(f"Weighted(adv trio) OOF: {score3:.6f} [{tag3}]")
    else:
        tag3, test3 = None, None

    # Pick best by OOF
    candidates = [
        ("stack_lr", score_lr, pred_lr),
        ("rank_avg_all", score_ra, test_rank_avg),
    ]
    if tag3 is not None:
        candidates.append((tag3, score3, test3))

    best_name, best_score, best_pred = max(candidates, key=lambda x: x[1])
    print(f"BEST by OOF -> {best_name}: {best_score:.6f}")

    # Save all variants
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"adv_ens_{ts}"
    # save best
    pd.DataFrame({"id": test_ids, "y": best_pred}).to_csv(
        SUB / f"{base}_{best_name}.csv", index=False
    )
    # also save others for quick LB probing
    pd.DataFrame({"id": test_ids, "y": pred_lr}).to_csv(
        SUB / f"{base}_stack_lr.csv", index=False
    )
    pd.DataFrame({"id": test_ids, "y": test_rank_avg}).to_csv(
        SUB / f"{base}_rank_avg_all.csv", index=False
    )
    if tag3 is not None:
        pd.DataFrame({"id": test_ids, "y": test3}).to_csv(
            SUB / f"{base}_{tag3}.csv", index=False
        )

    print("Saved submissions to:", SUB)
    print("Done.")


if __name__ == "__main__":
    main()
