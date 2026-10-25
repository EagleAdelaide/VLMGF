#!/usr/bin/env python3
"""CPU smoke test (seconds). Part 1 recomputes nine-model AUROCs from shipped per-row scores;
parts 2-3 only read saved summary tables. For recomputation and refits use repro/reproduce.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / "data_compact"

def rank_auc(score_fail, y_fail):
    r = pd.Series(score_fail).rank(method="average").to_numpy()
    pos = np.asarray(y_fail) == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

# 1) nine-model macro AUROCs from per-row files
vals = {"full_logistic": [], "full_gbdt": [], "geom_gbdt": []}
for f in sorted(D.glob("per_row_*.parquet")):
    if f.name == "per_row_e2.parquet":
        continue
    df = pd.read_parquet(f)
    for m in vals:
        vals[m].append(rank_auc(1 - df[f"prob_success_{m}"], df["y_fail"]))
macro = {m: round(float(np.mean(v)), 4) for m, v in vals.items()}
exp = {"full_logistic": 0.7034, "full_gbdt": 0.7482, "geom_gbdt": 0.7031}
print("nine-model macro AUROC:", macro)
assert all(abs(macro[m] - exp[m]) < 5e-4 for m in exp), (macro, exp)

# 2) stable factor categories -- reads the saved gate table (refit: reproduce.py --level L2)
g = pd.read_csv(D / "factor_stability_rev1.csv")
stable = sorted(g.loc[g["stable_rev_gate"], "category"].unique())
print("stable categories (saved table):", stable)
assert len(stable) == 7, stable

# 3) Web held-out metrics -- reads the saved summary (recomputed from per-row scores in reproduce.py)
w = pd.read_csv(D / "e3_web" / "point_metrics_web.csv")
tw = w[(w["split"] == "test_website") & (w["outcome"] == "click_fail") &
       (w["method"] == "geom_token_logistic")]
for _, r in tw.iterrows():
    print(f"web {r['model']} (saved table): AUROC {r['auroc']:.3f} risk@50 {r['risk_at_50']:.3f}")
q = tw[tw["model"] == "qwen3vl4b"].iloc[0]
assert abs(q["auroc"] - 0.710) < 5e-3 and abs(q["risk_at_50"] - 0.106) < 5e-3
print("QUICKCHECK OK")
