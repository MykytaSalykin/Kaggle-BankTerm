# Bank Term Deposit — Binary Classification (Kaggle)

**Goal:** predict whether a client will subscribe to a term deposit after a marketing contact.  
**Metric:** ROC–AUC (higher is better)

This repo provides a clean, CPU-friendly pipeline that is:
- **Reproducible** — deterministic CV and cached OOF/test predictions
- **Practical** — fast to train locally, simple to extend
- **Competitive** — strong single models + a simple stacker

---

## 🔍 What’s inside

- **Feature set**  
  Curated, competition-proven transforms:
  - clipped and log-scaled `duration`, `pdays` handling (`-1` → “not contacted” flag + log on positives)
  - per-call intensity `duration_per_call`
  - cyclic encodings for `month`/`day`
  - simple interactions (e.g. `duration × cellular`)
  - careful dtype management for native categorical support

- **Models**
  - **LightGBM (GBDT):** 3 seeds + a `scale_pos_weight` variant  
  - **CatBoost (CPU)** with native categorical handling  
  - **XGBoost (hist)** with one-hot for categoricals (via `OneHotEncoder`)
  - **Meta-learner:** Logistic Regression stacking on OOFs

- **Cross-validation**  
  `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`  
  OOF predictions are cached to `notebooks/outputs/cache/`.

---

## 📊 Current results (5-fold OOF AUC)

| Model               | OOF AUC  |
|---------------------|----------|
| LightGBM (seed=42)  | **0.97103** |
| LightGBM (seed=7)   | **0.97106** |
| LightGBM (+pos_w)   | 0.97060  |
| CatBoost (CPU)      | 0.96733  |
| XGBoost (OHE)       | 0.96922  |
| **Stack (LR on OOF)** | **0.97091** |

> Notes:
> - LightGBM singles are the strongest; stack is intentionally simple and transparent.
> - Scores are obtained on the provided `data/train.csv` with the feature set in `02_baselines.ipynb`.

---

### Public leaderboard (ensembling in `score_lab/`)

| Submission                  | LB Score | Notes                         |
|-----------------------------|----------|-------------------------------|
| **A2 mean-rank all**        | **0.97678** | Best public score (The 10% best) |
| Geometric mean (all models) | 0.97677  | Very close to best            |
| Diverse mean-rank top6      | 0.97636  | Slightly worse                |
| Weighted blends (C1)        | ~0.972   | Overfit, dropped              |

> ✅ Best LB score: **0.97678**.  
> Stacking + blind blending explored in `score_lab/`, but baselines remain clean and interpretable.


---

## 🚀 Quick start

1) **Data**  
Place files into `./data/`:
data/
├─ train.csv
└─ test.csv

2) **Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3) **Reproduce baselines (recommended)**
Open notebooks/02_baselines.ipynb
Run all cells → this will train models and populate:
notebooks/outputs/cache/*_oof.npy
notebooks/outputs/cache/*_test.npy
notebooks/outputs/submissions/final_stack_*.csv

If cache already exists, you can regenerate only the final submission
```bash
python run_stack.py
```
Outputs appear in notebooks/outputs/submissions/.


🧠 Why this setup

Determinism & speed. 
Everything fits on CPU and reproduces exactly thanks to fixed seeds and cached OOF.

Clarity > complexity. 
Strong baselines with clean feature logic beat fragile over-tuned stacks in most tabular comps.

Extendability. 
Add a new model? Just drop its OOF/TEST into cache and it instantly plugs into the stacker.


## 📁 Project structure
├── data/                          # train/test CSVs (not tracked)
├── notebooks/
│   ├── 01_eda.ipynb              # clean EDA with checks/drift/mutual info
│   ├── 02_baselines.ipynb        # features + models + caching + stack
│   └── outputs/
│       ├── cache/                # .npy OOF/TEST files
│       └── submissions/          # final CSVs for Kaggle
├── score_lab/                    # extra ensembling "for leaderboard only"
│   ├── pool/                     # external/public submissions
│   ├── ensemble.py               # blending recipes
│   └── notes.md                  # LB tracking
├── requirements.txt
├── run_stack.py                  # builds final stack from the cache
└── README.md


📜 License
MIT License — free to use and adapt.