#!/usr/bin/env python3
"""Same-accepted-set outcome replay and task-cluster risk-coverage bands (2026-10-20, CPU only).

(1) For each model, apply the calibration-frozen geometry+token logistic threshold (median failure
    score on the 200 calibration actions; accept iff failure score <= median) to the 852 test_website
    actions ONCE, and score the SAME accepted IDs under both outcomes: O2 point-in-target failure
    (click_fail) and O1 IoU failure (iou_fail). Task-cluster bootstrap intervals (2,000 draws).
(2) Risk-vs-coverage curves with task-cluster bootstrap bands: for each draw, resample tasks with
    replacement, rebuild the curve on the resampled rows, read it on a fixed coverage grid; report
    2.5/97.5 percentiles. The point curve is the observed one (no resampling).
(3) 500-action sampling-panel summary for the cost/performance figure (from the frozen R9 summaries).

Usage: python p0_accepted_set_and_bands_2026_10_20.py <artifact_root> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from entropy_scale import load_scores, failure_score, correction_root

SEED, B = 20261009, 2000
MODELS = ["qwen3vl4b", "internvl35_4b"]
HEADS = ["raw_coord_mean_entropy", "token_logistic", "geom_token_logistic", "geom_gbdt"]
GRID = np.round(np.arange(0.05, 1.0001, 0.01), 2)


def curve_on_grid(fail_score, y, grid):
    order = np.argsort(fail_score, kind="stable")
    cum = np.cumsum(y[order]); k = np.arange(1, len(y) + 1)
    risk = cum / k; cov = k / len(y)
    return np.interp(grid, cov, risk)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    acc_rows, band_rows, ids_rows = [], [], []
    for m in MODELS:
        sc = load_scores(root, m)
        cal = sc[sc.split == "calib"]; tw = sc[sc.split == "test_website"].reset_index(drop=True)
        assert len(cal) == 200 and len(tw) == 852 and tw.sample_id.is_unique
        col = "prob_success_geom_token_logistic"
        tau = float(np.quantile(1 - cal[col], 0.5))
        acc = ((1 - tw[col]) <= tau).to_numpy()
        A = tw[acc]
        ids_rows += [dict(model=m, sample_id=s) for s in A.sample_id]
        tasks = tw.task.to_numpy(); uniq = np.unique(tasks); idx = {t: np.flatnonzero(tasks == t) for t in uniq}
        o2, o1, cov = [], [], []
        y2, y1 = tw.click_fail.to_numpy(), tw.iou_fail.to_numpy()
        for _ in range(B):
            ii = np.concatenate([idx[t] for t in rng.choice(uniq, size=len(uniq), replace=True)])
            a = acc[ii]
            if a.sum() == 0:
                continue
            o2.append(y2[ii][a].mean()); o1.append(y1[ii][a].mean()); cov.append(a.mean())
        q = lambda v: (float(np.quantile(v, .025)), float(np.quantile(v, .975)))
        acc_rows.append(dict(model=m, rule="accept iff (1-p_success_geom_token_logistic) <= calibration median", tau=round(tau, 6),
                             n_test=852, n_accepted=int(acc.sum()), coverage=round(acc.mean(), 4),
                             o2_point_fail_rate=round(float(A.click_fail.mean()), 4), o2_lo=round(q(o2)[0], 4), o2_hi=round(q(o2)[1], 4),
                             o1_iou_fail_rate=round(float(A.iou_fail.mean()), 4), o1_lo=round(q(o1)[0], 4), o1_hi=round(q(o1)[1], 4),
                             cov_lo=round(q(cov)[0], 4), cov_hi=round(q(cov)[1], 4),
                             n_o2_fail=int(A.click_fail.sum()), n_o1_fail=int(A.iou_fail.sum()),
                             n_iou_fail_but_point_hit_in_accepted=int(((A.iou_fail == 1) & (A.click_fail == 0)).sum()), boot_draws=len(o2)))
        # risk-coverage bands per head; also a random-order reference (expected risk = prevalence)
        for h in HEADS:
            f = failure_score(tw,h).to_numpy()
            obs = curve_on_grid(f, y2, GRID)
            draws = []
            for _ in range(B):
                ii = np.concatenate([idx[t] for t in rng.choice(uniq, size=len(uniq), replace=True)])
                draws.append(curve_on_grid(f[ii], y2[ii], GRID))
            draws = np.array(draws)
            lo, hi = np.quantile(draws, .025, axis=0), np.quantile(draws, .975, axis=0)
            for g, o, l, u in zip(GRID, obs, lo, hi):
                band_rows.append(dict(model=m, head=h, coverage=float(g), risk=round(float(o), 5), lo=round(float(l), 5), hi=round(float(u), 5)))
    pd.DataFrame(acc_rows).to_csv(out / "p0b_same_accepted_set_two_outcomes.csv", index=False)
    pd.DataFrame(ids_rows).to_csv(out / "p0b_accepted_ids_geom_token_frozen.csv", index=False)
    pd.DataFrame(band_rows).to_csv(out / "p0c_risk_coverage_bands.csv", index=False)
    # sampling-panel summary for the cost figure (frozen R9 summaries; 500 test actions; first 500 of test_website)
    rows = []
    for m, p, base_s, extra_s in [("qwen3vl4b", "r9_1", 1.4934155, 8.2637145), ("internvl35_4b", "r9_1/internvl35_4b", 1.4224428, 3.1451124)]:
        s = json.load(open(correction_root(root) / ("r9_"+m) / "eval/r9_summary.json"))
        for key, lab, reg, fit, gens in [("geom_gbdt", "Geometry GBDT", "A", 510, 1), ("full_logistic", "Full observables logistic", "A", 510, 1),
                                          ("raw_coord_entropy", "Coord. entropy (raw)", "B", 0, 1), ("token_only", "Token-only logistic", "B", 510, 1),
                                          ("geom_token", "Geometry+token logistic", "B", 510, 1),
                                          ("sg_code_default", "Dispersion K=10 (code params)", "C", 0, 11), ("sg_paper_params", "Dispersion K=10 (paper params)", "C", 0, 11)]:
            pm = s["point_metrics"][key]; ci = s["auroc_ci"][key]
            rows.append(dict(model=m, score=key, label=lab, regime=reg, fit_labels=fit, threshold_inputs=200, generations=gens,
                             latency_s=round(base_s + (extra_s if gens > 1 else 0.0), 3), auroc=round(pm["auroc"], 4), auroc_lo=round(ci[0], 4), auroc_hi=round(ci[1], 4),
                             n=s["n_test"], n_fail=int(round(s["test_failure_rate"] * s["n_test"]))))
    pd.DataFrame(rows).to_csv(out / "p0c_sampling_panel_cost_points.csv", index=False)
    print(pd.DataFrame(acc_rows).to_string())


if __name__ == "__main__":
    main()
