from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import xgboost as xgb

from advanced.common import prepare_frames, ensure_outputs, save_cache


def _le_df_pair(X, Xte):
    Xl, Tl = X.copy(), Xte.copy()
    for c in Xl.select_dtypes(include=["category", "object"]).columns:
        vals = pd.Categorical(
            pd.concat([Xl[c].astype(str), Tl[c].astype(str)], ignore_index=True)
        )
        Xl[c] = vals[: len(Xl)].codes
        Tl[c] = vals[len(Xl) :].codes
    return Xl, Tl


def train_xgb(use_orig=True, seed=42):
    OUT, CACHE, SUB = ensure_outputs()
    X, y, Xte, used_orig = prepare_frames(use_orig=use_orig)
    Xl, Xtel = _le_df_pair(X, Xte)
    print(f"X (LE): {Xl.shape}, Xte (LE): {Xtel.shape}")
    print("Used bank-full.csv:", bool(used_orig))

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.02,
        "max_leaves": 127,
        "max_depth": 0,
        "grow_policy": "lossguide",
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(Xl))
    pred = np.zeros(len(Xtel))
    for fold, (tr, va) in enumerate(
        tqdm(skf.split(Xl, y), total=skf.n_splits, desc="XGBoost CV"), 1
    ):
        print(f"[XGB] Fold {fold}/{skf.n_splits}")
        dtr = xgb.DMatrix(Xl.iloc[tr], label=y[tr])
        dva = xgb.DMatrix(Xl.iloc[va], label=y[va])
        dte = xgb.DMatrix(Xtel)
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=6000,
            evals=[(dtr, "train"), (dva, "valid")],
            callbacks=[
                xgb.callback.EarlyStopping(rounds=400, save_best=True, maximize=True),
                xgb.callback.EvaluationMonitor(show_stdv=False),
            ],
        )
        oof[va] = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        pred += (
            bst.predict(dte, iteration_range=(0, bst.best_iteration + 1)) / skf.n_splits
        )

    print(f"XGB-hist OOF AUC: {roc_auc_score(y, oof):.6f}")
    save_cache(CACHE, "adv_xgb_hist_orig", oof, pred)
    print(f"Saved cache: {CACHE}/adv_xgb_hist_orig_*.npy")


if __name__ == "__main__":
    train_xgb(use_orig=True, seed=42)
