# score_lab/ensemble.py
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
from scipy.stats import rankdata, spearmanr

POOL = Path("score_lab/pool")
OUT = Path("score_lab/final")
OUT.mkdir(parents=True, exist_ok=True)


def load_pool():
    files = sorted(POOL.glob("*.csv"))
    assert files, "Put submission CSVs into score_lab/pool/ first."
    dfs = []
    base_id = None
    for f in files:
        df = pd.read_csv(f)
        if "id" not in df.columns or "y" not in df.columns:
            raise ValueError(f"{f.name} must have columns: id,y")
        df = df[["id", "y"]].copy()
        if base_id is None:
            base_id = df["id"].values
        else:
            if not np.array_equal(base_id, df["id"].values):
                raise ValueError(f"id order mismatch in {f.name}")
        df.rename(columns={"y": f.stem}, inplace=True)
        dfs.append(df.set_index("id"))
    pool = pd.concat(dfs, axis=1)
    return pool


def to_rank01(col):
    r = rankdata(col, method="average")
    return (r - 1) / (len(col) - 1)


def save_pred(y, tag):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT / f"{ts}_{tag}.csv"
    # recover id index from any pool df we pass in
    if isinstance(y, pd.Series):
        sub = pd.DataFrame({"id": y.index, "y": y.values})
    else:
        raise ValueError("y must be a pandas Series with id index")
    sub.to_csv(path, index=False)
    print("Saved:", path.name)


def main():
    pool = load_pool()  # index=id, columns=names
    print(f"Loaded {pool.shape[1]} submissions, n_rows={len(pool)}")

    # summary correlations
    cols = list(pool.columns)
    corr = np.zeros((len(cols), len(cols)))
    for i in range(len(cols)):
        for j in range(len(cols)):
            corr[i, j] = spearmanr(pool.iloc[:, i], pool.iloc[:, j]).correlation
    avg_corr = corr.mean(axis=1)
    order = np.argsort(avg_corr)  # from most diverse to most similar
    print("\n=== Avg Spearman per file (lower -> more diverse) ===")
    for i in order:
        print(f"{cols[i]:35s}  avg_r={avg_corr[i]:.6f}")

    # Normalizations
    ranked = pool.apply(to_rank01, axis=0)  # rank-space [0,1]
    clipped = pool.clip(0, 1)

    # A) Simple blends over ALL models
    mean_raw = clipped.mean(axis=1)
    mean_rank = ranked.mean(axis=1)
    geom_rank = np.exp(np.log(ranked.clip(1e-9)).mean(axis=1))
    trim_rank = ranked.apply(np.sort, axis=1, result_type="broadcast")
    k = max(1, int(0.1 * ranked.shape[1]))
    trimmed = (
        trim_rank.iloc[:, k : ranked.shape[1] - k].mean(axis=1)
        if ranked.shape[1] > 2 * k
        else mean_rank
    )

    save_pred(mean_raw, "A1_mean_raw_all")
    save_pred(mean_rank, "A2_mean_rank_all")
    save_pred(geom_rank, "A3_geom_rank_all")
    save_pred(trimmed, f"A4_trim{int(100 * k / ranked.shape[1])}_rank_all")

    # B) Focus subsets: pick 3–6 most diverse
    diverse_cols = [cols[i] for i in order[: min(6, len(cols))]]
    focus_rank = ranked[diverse_cols].mean(axis=1)
    save_pred(focus_rank, f"B1_mean_rank_diverse_top{len(diverse_cols)}")

    # C) Small weight grid on 3 least-correlated among top-K
    K = min(6, len(cols))
    best = None
    best_tuple = None
    # choose subset of size 3 from the K most diverse
    picks = [cols[i] for i in order[:K]]
    for a, b, c in combinations(picks, 3):
        R = ranked[[a, b, c]]
        # sweep weights in coarse grid that sums to 1
        for wa in np.linspace(0.2, 0.6, 9):
            for wb in np.linspace(0.2, 0.6, 9):
                wc = 1.0 - wa - wb
                if wc < 0.0:
                    continue
                blend = wa * R[a] + wb * R[b] + wc * R[c]
                # diversity proxy: lower avg corr of blend to components is better
                proxy = np.mean([spearmanr(blend, R[k]).correlation for k in [a, b, c]])
                if (best is None) or (proxy < best):
                    best = proxy
                    best_tuple = (a, b, c, wa, wb, wc, blend)
    if best_tuple is not None:
        a, b, c, wa, wb, wc, blend = best_tuple
        tag = f"C1_weighted_rank_{Path(a).stem}_{Path(b).stem}_{Path(c).stem}_{wa:.2f}_{wb:.2f}_{wc:.2f}"
        save_pred(blend, tag)

    # D) Power-mean in rank space (p<1 emphasizes small probs)
    for p in [0.5, 0.7, 1.5, 2.0]:
        pm = ((ranked**p).mean(axis=1)) ** (1.0 / p)
        save_pred(pm, f"D_pm_rank_p{p}")

    print("\nDone. Submit A2_mean_rank_all first; then B1_* and C1_*.")


if __name__ == "__main__":
    main()
