from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import lightgbm as lgb

from advanced.common import prepare_frames, ensure_outputs, save_cache


def train_lgbm():
    OUT, CACHE, SUB = ensure_outputs()
    X, y, Xte, used_orig = prepare_frames(use_orig=True)  # <-- important
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    print(f"X: {X.shape}, Xte: {Xte.shape}")
    print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")
    print("Used bank-full.csv:", bool(used_orig))

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 127,
        "min_data_in_leaf": 96,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "max_bin": 511,
        "verbosity": -1,
        "seed": 42,
        "force_row_wise": True,
        "n_jobs": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    pred = np.zeros(len(Xte))
    for fold, (tr, va) in enumerate(
        tqdm(skf.split(X, y), total=skf.n_splits, desc="LightGBM CV"), 1
    ):
        print(f"[LGBM] Fold {fold}/{skf.n_splits}")
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        dtr = lgb.Dataset(
            Xtr,
            label=ytr,
            categorical_feature=[c for c in cat_cols if c in X.columns],
            free_raw_data=False,
        )
        dva = lgb.Dataset(
            Xva,
            label=yva,
            categorical_feature=[c for c in cat_cols if c in X.columns],
            free_raw_data=False,
        )
        m = lgb.train(
            params,
            dtr,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            num_boost_round=4000,
            callbacks=[lgb.early_stopping(350), lgb.log_evaluation(200)],
        )
        best_iter = m.best_iteration
        oof[va] = m.predict(Xva, num_iteration=best_iter)
        pred += m.predict(Xte, num_iteration=best_iter) / skf.n_splits

    print(f"LGBM-GBDT OOF AUC: {roc_auc_score(y, oof):.6f}")
    save_cache(CACHE, "adv_lgb_gbdt_orig", oof, pred)
    print(f"Saved cache: {CACHE}/adv_lgb_gbdt_orig_*.npy")


if __name__ == "__main__":
    train_lgbm()
