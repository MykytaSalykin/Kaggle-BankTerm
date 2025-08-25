from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from catboost import CatBoostClassifier, Pool

from advanced.common import prepare_frames, ensure_outputs, save_cache


def train_catboost():
    OUT, CACHE, SUB = ensure_outputs()
    X, y, Xte, used_orig = prepare_frames(use_orig=True)  # <-- important
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    print(f"X: {X.shape}, Xte: {Xte.shape}")
    print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")
    print("Used bank-full.csv:", bool(used_orig))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    pred = np.zeros(len(Xte))
    for fold, (tr, va) in enumerate(
        tqdm(skf.split(X, y), total=skf.n_splits, desc="CatBoost CV"), 1
    ):
        print(f"[CAT] Fold {fold}/{skf.n_splits}")
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        trp = Pool(Xtr, ytr, cat_features=cat_idx)
        vap = Pool(Xva, yva, cat_features=cat_idx)
        tep = Pool(Xte, cat_features=cat_idx)
        m = CatBoostClassifier(
            iterations=3500,
            depth=6,
            learning_rate=0.045,
            l2_leaf_reg=6,
            loss_function="Logloss",
            eval_metric="AUC",
            od_type="Iter",
            od_wait=300,
            random_seed=42 + fold,
            allow_writing_files=False,
            verbose=200,
            task_type="CPU",
        )
        m.fit(trp, eval_set=vap, use_best_model=True)
        oof[va] = m.predict_proba(vap)[:, 1]
        pred += m.predict_proba(tep)[:, 1] / skf.n_splits

    print(f"CatBoost OOF AUC: {roc_auc_score(y, oof):.6f}")
    save_cache(CACHE, "adv_cat_orig", oof, pred)
    print(f"Saved cache: {CACHE}/adv_cat_orig_*.npy")


if __name__ == "__main__":
    train_catboost()
