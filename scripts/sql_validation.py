#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQL-style data validation for Bank Term dataset using DuckDB.
Run:
    python scripts/sql_validation.py
"""

from pathlib import Path
import duckdb

# ---------- paths ----------
PROJ = Path(__file__).resolve().parents[1]
DATA = None
for p in [PROJ / "data", PROJ.parent / "data"]:
    if (p / "train.csv").exists():
        DATA = p
        break
if DATA is None:
    raise FileNotFoundError("data/train.csv not found (looked in ./data and ../data)")

train_path = str((DATA / "train.csv").resolve())

# ---------- connect ----------
con = duckdb.connect()

# ---------- create a typed view ----------
# DuckDB infers types well, but we do that explicitly for clarity:
# "default", "housing", "loan" → BOOLEAN (yes/no), others — strings/numbers.
con.execute(f"""
CREATE OR REPLACE VIEW bank AS
SELECT
    CAST(id AS BIGINT)                      AS id,
    CAST(age AS INTEGER)                    AS age,
    job::VARCHAR                            AS job,
    marital::VARCHAR                        AS marital,
    education::VARCHAR                      AS education,
    CASE WHEN lower(CAST("default" AS VARCHAR)) = 'yes' THEN TRUE ELSE FALSE END AS has_default,
    CAST(balance AS BIGINT)                 AS balance,
    CASE WHEN lower(CAST(housing AS VARCHAR)) = 'yes' THEN TRUE ELSE FALSE END AS has_housing,
    CASE WHEN lower(CAST(loan    AS VARCHAR)) = 'yes' THEN TRUE ELSE FALSE END AS has_loan,
    contact::VARCHAR                        AS contact,
    CAST(day AS INTEGER)                    AS day,
    month::VARCHAR                          AS month,
    CAST(duration AS INTEGER)               AS duration,
    CAST(campaign AS INTEGER)               AS campaign,
    CAST(pdays AS INTEGER)                  AS pdays,
    CAST(previous AS INTEGER)               AS previous,
    poutcome::VARCHAR                       AS poutcome,
    CAST(y AS INTEGER)                      AS y
FROM read_csv_auto('{train_path}', header=True);
""")

# ---------- discover schema ----------
cols_info = con.execute("PRAGMA table_info('bank');").fetchdf()
cols = cols_info["name"].tolist()

num_cols = [
    c
    for c in cols
    if con.execute(f"SELECT typeof({c}) FROM bank LIMIT 1;").fetchone()[0]
    in ("INTEGER", "BIGINT", "DOUBLE", "HUGEINT")
]
cat_cols = [c for c in cols if c not in num_cols and c != "y"]

# ---------- row count ----------
row_cnt = con.execute("SELECT COUNT(*) FROM bank;").fetchone()[0]

# ---------- nulls per column ----------
null_exprs = [
    f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}_nulls" for c in cols
]
nulls_sql = "SELECT " + ", ".join(null_exprs) + " FROM bank;"
nulls_row = con.execute(nulls_sql).fetchdf().T
nulls_row.columns = ["nulls"]
nulls_row["column"] = nulls_row.index
nulls_row = nulls_row[["column", "nulls"]]

# ---------- duplicates by id ----------
dup_id = con.execute("""
SELECT COUNT(*) - COUNT(DISTINCT id) AS duplicate_id_rows FROM bank;
""").fetchone()[0]

# ---------- categorical cardinalities ----------
card_rows = []
for c in cat_cols:
    n = con.execute(f"SELECT COUNT(DISTINCT {c}) FROM bank;").fetchone()[0]
    card_rows.append((c, int(n)))

# ---------- numeric summary ----------
num_summary = None
if num_cols:
    num_sql = (
        "SELECT "
        + ", ".join(
            [
                f"MIN({c}) AS {c}_min, AVG({c}) AS {c}_mean, STDDEV_POP({c}) AS {c}_std, MAX({c}) AS {c}_max"
                for c in num_cols
            ]
        )
        + " FROM bank;"
    )
    num_summary = con.execute(num_sql).fetchdf().T
    num_summary.columns = ["value"]
    num_summary["metric"] = num_summary.index
    num_summary = num_summary[["metric", "value"]]

# ---------- target rate & scale_pos_weight ----------
y_rate, spw = con.execute("""
WITH t AS (
  SELECT SUM(CASE WHEN y=1 THEN 1 ELSE 0 END) AS pos,
         SUM(CASE WHEN y=0 THEN 1 ELSE 0 END) AS neg
  FROM bank
)
SELECT CAST(pos AS DOUBLE)/(pos+neg) AS positive_rate,
       CASE WHEN pos=0 THEN NULL ELSE CAST(neg AS DOUBLE)/pos END AS scale_pos_weight
FROM t;
""").fetchone()

# ---------- print report ----------
print("\n=== BANK TERM — SQL VALIDATION REPORT ===")
print(f"Rows: {row_cnt:,}")
print(f"Duplicate rows by id: {dup_id:,}")

print("\n-- Nulls per column --")
for _, row in nulls_row.iterrows():
    if int(row["nulls"]) != 0:
        print(f"{row['column']}: {int(row['nulls']):,}")
if all(int(v) == 0 for v in nulls_row["nulls"].tolist()):
    print("No NULLs detected in any column.")

print("\n-- Categorical cardinalities --")
for c, n in card_rows:
    print(f"{c}: {n}")

if num_summary is not None:
    print("\n-- Numeric summary (min / mean / std / max) --")
    for c in num_cols:
        mn, mean, std, mx = con.execute(
            f"SELECT MIN({c}), AVG({c}), STDDEV_POP({c}), MAX({c}) FROM bank;"
        ).fetchone()
        print(f"{c:<9}  min={mn:.0f}  mean={mean:.2f}  std={std:.2f}  max={mx:.0f}")

print("\n-- Target distribution & weighting --")
print(f"Positive class rate: {y_rate:.4f}")
print(f"Suggested scale_pos_weight: {spw:.3f}")

print("\nDone.")
