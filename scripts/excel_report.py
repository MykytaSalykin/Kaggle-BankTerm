# scripts/excel_report.py
import pandas as pd, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "docs" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

dd = pd.DataFrame(
    {
        "column": train.columns,
        "dtype": [str(t) for t in train.dtypes],
        "n_unique": [train[c].nunique() for c in train.columns],
        "has_nulls": [train[c].isna().any() for c in train.columns],
    }
)

p_job = train.pivot_table(index="job", values="y", aggfunc="mean").sort_values(
    "y", ascending=False
)
p_mon = train.pivot_table(index="month", values="y", aggfunc="mean").sort_values(
    "y", ascending=False
)
p_cont = train.pivot_table(index="contact", values="y", aggfunc="mean").sort_values(
    "y", ascending=False
)

q = pd.qcut(train["duration"], q=20, duplicates="drop")
b_duration = (
    train.groupby(q, observed=True)["y"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "y_mean", "count": "n"})
)

xlsx_path = OUT / "explainer_report.xlsx"
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xl:
    dd.to_excel(xl, sheet_name="Data Dictionary", index=False)
    p_job.to_excel(xl, sheet_name="Target by Job")
    p_mon.to_excel(xl, sheet_name="Target by Month")
    p_cont.to_excel(xl, sheet_name="Target by Contact")
    b_duration.to_excel(xl, sheet_name="Duration Bins")

print(f"[OK] Excel report -> {xlsx_path}")
