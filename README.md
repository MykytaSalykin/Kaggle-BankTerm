# Bank Term Deposit – Binary Classification (Kaggle)
- Metric: ROC-AUC
- CV: StratifiedKFold(5, shuffle=True, random_state=42)
- Models: LightGBM, CatBoost, XGBoost (OHE), stacking LR
- Reproduce: put train.csv/test.csv into data/, run notebooks/02_baselines.ipynb
- Submissions: outputs/submissions/
