# Bank Term Deposit — Binary Classification (Kaggle)

- **Metric:** ROC–AUC  
- **CV:** StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
- **Baseline models:** LightGBM (3 seeds + pos_weight), CatBoost (CPU), XGBoost (OHE). Meta-learner: Logistic Regression (stacking).  
- **Hardware:** CPU-only friendly.  

## Quick start
1. Place `train.csv` and `test.csv` into `data/`.
2. (Optional) Open `notebooks/02_baselines.ipynb` and run; this trains all base models and fills `notebooks/outputs/cache/`.
3. Or if cache is already there, run:
   ```bash
   python run_stack.py

Output CSV will be saved to notebooks/outputs/submissions/


## Notes
Categorical handling: native categories in LightGBM and CatBoost; one-hot for XGBoost.

Feature set: curated interactions and clipping for duration, cyclic encodings for month/day, plus a few boolean flags.

Stacking: 5-fold out-of-fold predictions of base models are used to train a logistic regression meta-model.