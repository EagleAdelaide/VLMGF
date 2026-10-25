#!/usr/bin/env python3
"""P0/P1 CPU-only re-analysis of frozen Web outputs (2026-10-20 revision round).

Extension script: reads ONLY shipped artifact files (per-row scores, per-row features, manifest)
and the frozen R9 summaries; refits nothing; changes no population, split, label, or threshold.
All thresholds are the frozen calibration-median rule of the final-audit protocol
(accept iff failure score 1-p <= its calibration-set median, i.e. 50% calibration coverage).

Outputs (CSV/JSON) go to <out_dir>; every file is named in NEW_ANALYSIS_PROVENANCE.md.

Usage: python p0_web_reanalysis_2026_10_20.py <artifact_root> <out_dir>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from entropy_scale import load_scores, failure_score, correction_root

SEED = 20261009
B = 2000
MODELS = ["qwen3vl4b", "internvl35_4b"]
HEADS = {  # column suffix -> (regime, fit labels, threshold-setting scored inputs, needs logits, generations)
    "geom_logistic": ("A", 510, 200, "no", 1), "geom_gbdt": ("A", 510, 200, "no", 1),
    "full_logistic": ("A", 510, 200, "no", 1), "full_gbdt": ("A", 510, 200, "no", 1),
    "raw_coord_mean_entropy": ("B", 0, 200, "yes", 1), "token_logistic": ("B", 510, 200, "yes", 1),
    "token_gbdt": ("B", 510, 200, "yes", 1), "geom_token_logistic": ("B", 510, 200, "yes", 1),
    "geom_token_gbdt": ("B", 510, 200, "yes", 1)}


def rank_auc(score_fail, y):
    r = pd.Series(score_fail).rank(method="average").to_numpy()
    y = np.asarray(y) == 1
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def task_boot_auc(df, col, y, rng):
    """Task-cluster bootstrap 95% interval of AUROC (resample tasks with replacement)."""
    tasks = df["task"].to_numpy()
    uniq = np.unique(tasks)
    idx = {t: np.flatnonzero(tasks == t) for t in uniq}
    vals = []
    s = failure_score(df,col).to_numpy(); yy = df[y].to_numpy()  # failure score = 1 - P(success)
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[t] for t in pick])
        vals.append(rank_auc(s[ii], yy[ii]))
    vals = np.array(vals); vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), int(len(vals))


def frozen_tau(cal, col):
    return float(np.quantile(failure_score(cal,col), 0.5))


def cov_risk(sub, col, tau, y="click_fail"):
    acc = failure_score(sub,col) <= tau
    n_acc = int(acc.sum())
    return float(acc.mean()), (float(sub.loc[acc, y].mean()) if n_acc else float("nan")), n_acc


def ece(p_succ, y_fail, bins=10):
    p = np.asarray(p_succ, float); ys = 1 - np.asarray(y_fail)
    edges = np.linspace(0, 1, bins + 1); e = 0.0; rows = []
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - ys[m].mean())
            rows.append((edges[i], edges[i + 1], int(m.sum()), float(p[m].mean()), float(ys[m].mean())))
    return float(e), rows


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    feat = pd.read_parquet(root / "data_compact/e3_web/per_row_web.parquet")
    man = pd.read_csv(root / "data_compact/mind2web_frozen/manifest.csv")
    rng = np.random.default_rng(SEED)
    prov = {}

    # ---------- P0-B: outcome definitions, contingency, action-type composition ----------
    rows_ct, rows_op = [], []
    for m in MODELS:
        t = feat[(feat.model == m) & (feat.split == "test_website")].copy()
        assert len(t) == 852, len(t)
        ct = dict(model=m, n=len(t),
                  iou_fail=int(t.iou_fail.sum()), point_fail=int(t.click_fail.sum()),
                  iou_fail_point_hit=int(((t.iou_fail == 1) & (t.click_fail == 0)).sum()),
                  iou_ok_point_fail=int(((t.iou_fail == 0) & (t.click_fail == 1)).sum()),
                  both_fail=int(((t.iou_fail == 1) & (t.click_fail == 1)).sum()),
                  both_ok=int(((t.iou_fail == 0) & (t.click_fail == 0)).sum()))
        ct["iou_fail_point_hit_pct_of_852"] = round(100 * ct["iou_fail_point_hit"] / 852, 1)
        ct["iou_fail_point_hit_pct_of_iou_failures"] = round(100 * ct["iou_fail_point_hit"] / ct["iou_fail"], 1)
        rows_ct.append(ct)
        for op, g in t.groupby("op"):
            rows_op.append(dict(model=m, op=op, n=len(g), tasks=g.task.nunique(), point_fail_rate=round(g.click_fail.mean(), 3),
                                iou_fail_rate=round(g.iou_fail.mean(), 3),
                                iou_fail_point_hit=int(((g.iou_fail == 1) & (g.click_fail == 0)).sum())))
    pd.DataFrame(rows_ct).to_csv(out / "p0b_outcome_contingency_852.csv", index=False)
    pd.DataFrame(rows_op).to_csv(out / "p0b_action_type_composition.csv", index=False)
    comp = man[man.split == "test_website"].groupby("op").size().to_dict()
    prov["test_website_op_composition_manifest"] = comp

    # ---------- P0-B/C/D: per-head metrics overall, CLICK-only, by op, by site, by size ----------
    overall, click_only, by_op, by_site, by_size, loo_site = [], [], [], [], [], []
    for m in MODELS:
        sc = load_scores(root, m)
        cal = sc[sc.split == "calib"]; tw = sc[sc.split == "test_website"].copy(); td = sc[sc.split == "test_domain"].copy()
        assert len(cal) == 200 and len(tw) == 852 and len(td) == 504
        # size terciles on gt_area within the 852 test_website actions (same cut for both models: same actions)
        tw["size_tercile"] = pd.qcut(tw.gt_area, 3, labels=["small", "medium", "large"])
        for head, (reg, fit, calib, logit, gens) in HEADS.items():
            col = f"prob_success_{head}"
            tau = frozen_tau(cal, col)
            cal_cov = float((failure_score(cal,col) <= tau).mean())
            for name, sub, store in [("test_website_all", tw, overall), ("test_website_CLICK", tw[tw.op == "CLICK"], click_only),
                                     ("test_domain_all", td, overall)]:
                a = rank_auc(failure_score(sub,col), sub.click_fail)
                lo, hi, neff = task_boot_auc(sub, col, "click_fail", rng) if head in ("token_logistic", "geom_token_logistic", "raw_coord_mean_entropy", "geom_gbdt", "geom_token_gbdt") else (float("nan"), float("nan"), 0)
                cov, risk, nacc = cov_risk(sub, col, tau)
                k20 = int(math.ceil(0.2 * len(sub)))
                order = np.argsort(-(failure_score(sub,col).to_numpy()), kind="stable")
                c20 = int(sub.click_fail.to_numpy()[order[:k20]].sum())
                store.append(dict(model=m, population=name, head=head, regime=reg, fit_labels=fit, threshold_inputs=calib, needs_logits=logit,
                                  generations=gens, n=len(sub), n_fail=int(sub.click_fail.sum()), prevalence=round(sub.click_fail.mean(), 4),
                                  auroc=round(a, 4), auroc_lo=round(lo, 4), auroc_hi=round(hi, 4), boot_draws=neff, c20=c20, k20=k20,
                                  tau_failscore=round(tau, 6), calib_coverage=cal_cov, frozen_coverage=round(cov, 4),
                                  frozen_accepted_risk=(round(risk, 4) if nacc else "NA"), n_accepted=nacc))
            if head in ("token_logistic", "geom_token_logistic", "raw_coord_mean_entropy"):
                for op, g in tw.groupby("op"):
                    cov, risk, nacc = cov_risk(g, col, tau); a = rank_auc(failure_score(g,col), g.click_fail)
                    by_op.append(dict(model=m, head=head, op=op, n=len(g), n_fail=int(g.click_fail.sum()), prevalence=round(g.click_fail.mean(), 3),
                                      auroc=(round(a, 3) if np.isfinite(a) else "NA (one class)"), frozen_coverage=round(cov, 3),
                                      frozen_accepted_risk=(round(risk, 3) if nacc else "NA (0 accepted)"), n_accepted=nacc,
                                      note="TYPE/SELECT: localization proxy only (no typing/selecting executed)" if op != "CLICK" else ""))
                for sz, g in tw.groupby("size_tercile", observed=True):
                    cov, risk, nacc = cov_risk(g, col, tau); a = rank_auc(failure_score(g,col), g.click_fail)
                    by_size.append(dict(model=m, head=head, size_tercile=sz, n=len(g), n_fail=int(g.click_fail.sum()), prevalence=round(g.click_fail.mean(), 3),
                                        auroc=(round(a, 3) if np.isfinite(a) else "NA (one class)"), frozen_coverage=round(cov, 3),
                                        frozen_accepted_risk=(round(risk, 3) if nacc else "NA (0 accepted)"), n_accepted=nacc))
                for site, g in tw.groupby("website"):
                    cov, risk, nacc = cov_risk(g, col, tau); a = rank_auc(failure_score(g,col), g.click_fail)
                    by_site.append(dict(model=m, head=head, website=site, n=len(g), tasks=g.task.nunique(), n_fail=int(g.click_fail.sum()),
                                        prevalence=round(g.click_fail.mean(), 3), auroc=(round(a, 3) if np.isfinite(a) else "NA (one class)"),
                                        frozen_coverage=round(cov, 3), frozen_accepted_risk=(round(risk, 3) if nacc else "NA (0 accepted)"), n_accepted=nacc))
                # leave-one-website-out aggregates of frozen coverage/risk and AUROC (site heterogeneity, descriptive)
                aggs = []
                for site in tw.website.unique():
                    g = tw[tw.website != site]; cov, risk, _ = cov_risk(g, col, tau); aggs.append((cov, risk, rank_auc(failure_score(g,col), g.click_fail)))
                aggs = np.array(aggs)
                loo_site.append(dict(model=m, head=head, n_sites=len(aggs), cov_min=round(aggs[:, 0].min(), 3), cov_max=round(aggs[:, 0].max(), 3),
                                     risk_min=round(aggs[:, 1].min(), 3), risk_med=round(float(np.median(aggs[:, 1])), 3), risk_max=round(aggs[:, 1].max(), 3),
                                     auroc_min=round(aggs[:, 2].min(), 3), auroc_med=round(float(np.median(aggs[:, 2])), 3), auroc_max=round(aggs[:, 2].max(), 3)))
    pd.DataFrame(overall).to_csv(out / "p0c_heads_overall_and_domain.csv", index=False)
    pd.DataFrame(click_only).to_csv(out / "p0b_heads_click_only_670.csv", index=False)
    pd.DataFrame(by_op).to_csv(out / "p0d_by_action_type_frozen.csv", index=False)
    pd.DataFrame(by_size).to_csv(out / "p0d_by_size_tercile_frozen.csv", index=False)
    pd.DataFrame(by_site).to_csv(out / "p0d_model_by_website_frozen.csv", index=False)
    pd.DataFrame(loo_site).to_csv(out / "p0d_leave_one_website_out.csv", index=False)

    # ---------- P1-A: BH/FDR on the 11 factor terms; odds ratios ----------
    co = pd.read_csv(root / "data_compact/joint_coefficients_rev1.csv")
    st = pd.read_csv(root / "data_compact/factor_stability_rev1.csv")
    prim = co[(co.analysis == "rev_valid_success_primary") & (co.term.isin(st.term))].copy()
    assert len(prim) == 11, len(prim)
    p = prim.p_value.to_numpy(); order = np.argsort(p); n = len(p)
    q = np.empty(n); prev = 1.0
    for rank_i in range(n - 1, -1, -1):
        i = order[rank_i]; prev = min(prev, p[i] * n / (rank_i + 1)); q[i] = prev
    prim["bh_q"] = q; prim["bh_sig_05"] = q < 0.05
    prim["bonferroni_sig_05"] = p < 0.05 / n
    prim["odds_ratio"] = np.exp(prim.estimate); prim["or_lo"] = np.exp(prim.ci_low); prim["or_hi"] = np.exp(prim.ci_high)
    prim = prim.merge(st[["term", "category", "stable_rev_gate", "src_boot_excludes0", "lom_sign_agree", "lof_sign_agree"]], on="term")
    prim[["category", "term", "estimate", "ci_low", "ci_high", "odds_ratio", "or_lo", "or_hi", "p_value", "bh_q", "bh_sig_05", "bonferroni_sig_05",
          "stable_rev_gate", "src_boot_excludes0", "lom_sign_agree", "lof_sign_agree"]].to_csv(out / "p1a_factor_terms_fdr.csv", index=False)
    prov["p1a_stable_gate_count"] = int(st.stable_rev_gate.sum())
    prov["p1a_bh_sig_among_stable"] = int((prim.bh_sig_05 & prim.stable_rev_gate).sum())
    prov["p1a_bonf_sig_among_stable"] = int((prim.bonferroni_sig_05 & prim.stable_rev_gate).sum())

    # ---------- P1-A: probability-calibration diagnostics of the Web heads (fit-510 -> test, no refit) ----------
    calrows = []
    for m in MODELS:
        sc = load_scores(root, m)
        for split in ["test_website", "test_domain"]:
            t = sc[sc.split == split]
            for head in ["token_logistic", "geom_token_logistic", "geom_logistic", "geom_gbdt", "full_gbdt"]:
                col = f"prob_success_{head}"; ps = t[col].to_numpy(); yf = t.click_fail.to_numpy()
                brier = float(np.mean((ps - (1 - yf)) ** 2)); e, bins = ece(ps, yf)
                fit_y = feat[(feat.model == m) & (feat.split == "fit")].click_fail.to_numpy()
                assert len(fit_y) == 510
                fit_success = float((1 - fit_y).mean())
                base = float(np.mean((fit_success - (1 - yf)) ** 2))
                oracle = float(np.mean(((1 - yf).mean() - (1 - yf)) ** 2))
                calrows.append(dict(model=m, split=split, head=head, n=len(t), brier=round(brier, 4), brier_fit_constant=round(base, 4), brier_test_oracle=round(oracle, 4), fit_success_rate=fit_success,
                                    ece10=round(e, 4), mean_pred_success=round(float(ps.mean()), 3), observed_success=round(float(1 - yf.mean()), 3),
                                    bins=json.dumps([(round(a, 1), round(b, 1), n_, round(pm, 3), round(om, 3)) for a, b, n_, pm, om in bins])))
    pd.DataFrame(calrows).to_csv(out / "p1a_web_probability_calibration.csv", index=False)

    (out / "provenance.json").write_text(json.dumps(dict(seed=SEED, bootstrap_draws=B, artifact_root=str(root), **prov), indent=1))
    print("done ->", out)


if __name__ == "__main__":
    main()
