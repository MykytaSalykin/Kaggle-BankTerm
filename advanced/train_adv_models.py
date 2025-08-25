# advanced/train_adv_models.py
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from advanced.utils import (
    resolve_data_dir,
    find_orig_bank_full,
    ensure_outputs,
    save_cache,
)
from advanced.features import BASE_CAT, BASE_NUM, build_features
from tqdm.auto import tqdm


def load_core():
    data_dir = resolve_data_dir()
    assert data_dir is not None, "train.csv/test.csv not found. Put them into data/."
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test, data_dir


def load_bank_full():
    p = find_orig_bank_full()
    if p is None:
        return None
    sep = ";" if p.suffix == ".csv" else None
    df = pd.read_csv(p, sep=sep)  # official UCI/Kaggle version is ';'-separated
    if "y" in df.columns:
        df["y"] = df["y"].map({"no": 0, "yes": 1})
    return df


def prepare_frames():
    train, test, _ = load_core()
    orig = load_bank_full()

    y = train["y"].astype(int).copy()
    X = train.drop(columns=["id", "y"]).copy()
    X_test = test.drop(columns=["id"]).copy()

    if orig is not None:
        # ensure column alignment with Kaggle schema
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
        assert set(needed).issubset(set([c.lower() for c in orig.columns])), (
            "bank-full.csv columns mismatch"
        )
        orig = orig.rename(columns={c: c.lower() for c in orig.columns})
        orig = orig[needed].copy()
        X_orig = orig.drop(columns=["y"]).copy()
        y_orig = orig["y"].astype(int).copy()

        X_comb = pd.concat([X, X_orig], ignore_index=True)
        y_comb = pd.concat([y, y_orig], ignore_index=True)
    else:
        X_comb, y_comb = X, y

    X_feat, X_test_feat = build_features(X_comb, X_test, y=y_comb)
    return X_feat, y_comb.values, X_test_feat, (orig is not None)


def cv_catboost(X, y, Xte, cat_cols, seed=42):
    from catboost import CatBoostClassifier, Pool

    oof = np.zeros(len(X))
    pred = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(
        tqdm(skf.split(X, y), total=skf.n_splits, desc="CatBoost CV"), 1
    ):
        print("\n=== Training CatBoost ===")
        print(f"[CAT] Fold {fold}/{skf.n_splits}")
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        trp = Pool(
            Xtr,
            ytr,
            cat_features=[X.columns.get_loc(c) for c in cat_cols if c in X.columns],
        )
        vap = Pool(
            Xva,
            yva,
            cat_features=[X.columns.get_loc(c) for c in cat_cols if c in X.columns],
        )
        tep = Pool(
            Xte, cat_features=[X.columns.get_loc(c) for c in cat_cols if c in X.columns]
        )
        m = CatBoostClassifier(
            iterations=3000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=6,
            loss_function="Logloss",
            eval_metric="AUC",
            od_type="Iter",
            od_wait=200,
            random_seed=seed + fold,
            allow_writing_files=False,
            verbose=False,
        )
        m.fit(trp, eval_set=vap, use_best_model=True)
        oof[va] = m.predict_proba(vap)[:, 1]
        pred += m.predict_proba(tep)[:, 1] / skf.n_splits
    print("=== CatBoost done ===\n")
    return oof, pred


def cv_lgbm_dart(X, y, Xte, cat_cols, seed=42):
    import lightgbm as lgb

    oof = np.zeros(len(X))
    pred = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(
        tqdm(skf.split(X, y), total=skf.n_splits, desc="LightGBM CV"), 1
    ):
        print("\n=== Training LightGBM ===")
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
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "dart",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_data_in_leaf": 64,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 10.0,
            "max_bin": 511,
            "verbosity": -1,
            "seed": seed + fold,
            "force_row_wise": True,
            "n_jobs": -1,
        }
        cbs = [lgb.early_stopping(200), lgb.log_evaluation(200)]
        m = lgb.train(
            params,
            dtr,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            num_boost_round=5000,
            callbacks=cbs,
        )
        best_iter = m.best_iteration
        oof[va] = m.predict(Xva, num_iteration=best_iter)
        pred += m.predict(Xte, num_iteration=best_iter) / skf.n_splits
    print("=== LightGBM done ===\n")
    return oof, pred


def cv_xgb_hist(X, y, Xte, seed=42):
    from xgboost import XGBClassifier

    oof = np.zeros(len(X))
    pred = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # label-encode categories to integers for XGB
    X_le = X.copy()
    Xte_le = Xte.copy()
    for c in X_le.select_dtypes(include=["category", "object"]).columns:
        vals = pd.Categorical(
            pd.concat([X_le[c].astype(str), Xte_le[c].astype(str)], ignore_index=True)
        )
        X_le[c] = vals[: len(X_le)].codes
        Xte_le[c] = vals[len(X_le) :].codes

    for fold, (tr, va) in enumerate(
        tqdm(skf.split(X, y), total=skf.n_splits, desc="XGBoost CV"), 1
    ):
        print("\n=== Training XGB ===")
        print(f"[XGB] Fold {fold}/{skf.n_splits}")
        Xtr, Xva = X_le.iloc[tr], X_le.iloc[va]
        ytr, yva = y[tr], y[va]
        model = XGBClassifier(
            n_estimators=6000,
            learning_rate=0.02,
            max_leaves=127,
            max_depth=0,
            grow_policy="lossguide",
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=5.0,
            tree_method="hist",
            random_state=seed + fold,
            eval_metric="auc",
            n_jobs=-1,
        )
        model.fit(
            Xtr, ytr, eval_set=[(Xva, yva)], verbose=200, early_stopping_rounds=400
        )
        oof[va] = model.predict_proba(Xva)[:, 1]
        pred += model.predict_proba(Xte_le)[:, 1] / skf.n_splits
    print("=== XGBoost done ===\n")
    return oof, pred


if __name__ == "__main__":
    OUT, CACHE, SUB = ensure_outputs()
    X, y, Xte, used_orig = prepare_frames()

    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]

    oc, pc = cv_catboost(X, y, Xte, cat_cols, seed=42)
    print("CatBoost OOF:", roc_auc_score(y, oc))
    from advanced.utils import save_cache

    save_cache(CACHE, "adv_cat", oc, pc)

    ol, pl = cv_lgbm_dart(X, y, Xte, cat_cols, seed=42)
    print("LGBM-DART OOF:", roc_auc_score(y, ol))
    save_cache(CACHE, "adv_lgb_dart", ol, pl)

    ox, px = cv_xgb_hist(X, y, Xte, seed=42)
    print("XGB-hist OOF:", roc_auc_score(y, ox))
    save_cache(CACHE, "adv_xgb", ox, px)

    print("Used bank-full.csv:", used_orig)
