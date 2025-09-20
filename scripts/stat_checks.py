# scripts/stat_checks.py
from pathlib import Path
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

PROJ = Path(__file__).resolve().parents[1]
DATA = None
for p in [PROJ / "data", PROJ.parent / "data", PROJ.parent.parent / "data"]:
    if (p / "train.csv").exists() and (p / "test.csv").exists():
        DATA = p
        break
assert DATA, "data/train.csv and data/test.csv not found"

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
cat_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

rows = []
for c in num_cols:
    stat, p = ks_2samp(train[c].astype(float), test[c].astype(float))
    rows.append({"feature": c, "test": "KS", "statistic": stat, "p_value": p})
for c in cat_cols:
    ct = pd.concat(
        [
            train[c].value_counts().rename("train"),
            test[c].value_counts().rename("test"),
        ],
        axis=1,
    ).fillna(0)
    chi2, p, _, _ = chi2_contingency(ct)
    rows.append({"feature": c, "test": "Chi-square", "statistic": chi2, "p_value": p})

out = pd.DataFrame(rows).sort_values("p_value")
OUT = PROJ / "docs" / "validation"
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "stat_tests.csv", index=False)
print("[OK] Stat tests ->", OUT)
