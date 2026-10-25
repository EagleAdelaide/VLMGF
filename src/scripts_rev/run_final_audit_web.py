#!/usr/bin/env python3
"""Final-audit Web addendum (FINAL_AUDIT_PROTOCOL A-E). CPU only; fit on
fit-510 only; calib-200 only for frozen thresholds; test splits evaluation-only.

Outputs under www2027_revision/corrections/entropy_scale_v1/final_audit/:
  WEB_STRONG_BASELINES.csv      per model x split x method point metrics (all heads
                                incl. new token_gbdt / geom_token_gbdt) + fit/calib budgets
  per_row_scores_<model>.parquet  per-row scores for every head (both test splits + calib)
  WEB_RANDOM_BASELINE_MC.csv    MC (10,000 perms) mean + [2.5,97.5] + theoretical E
  WEB_DELTAS_BOOTSTRAP.json     paired synchronized task-group deltas (2,000 draws)
  WEB_SPLIT_COMPARISON.csv      website vs domain side-by-side
  WEB_SITE_SENSITIVITY.csv      per-website metrics + LOO-website aggregates
  WEB_ELIGIBILITY_AUDIT.csv     per-split exclusion ledger + op/site distributions
  WEB_SUBSET_SENSITIVITY.csv    S1/S2/S3 exploratory subsets
  WEB_NUMERIC_CONSISTENCY.json  key numbers for the paper
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("VLMGF_REPO", "."))
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "www2027_revision" / "scripts_rev"))
import run_e1a_baselines as E  # noqa: E402

OUT = REPO / "www2027_revision" / "corrections" / "entropy_scale_v1" / "final_audit"
OUT.mkdir(parents=True, exist_ok=True)
SEED_FIT = 20261009
SEED_MC = 20261010
NBOOT = 2000
NMC = 10000
TOKENF = ["coord_mean_logprob", "coord_min_logprob", "coord_mean_entropy", "out_mean_logprob"]
GEOM = ["pred_area_ratio", "pred_cx", "pred_cy", "pred_w", "pred_h", "pred_aspect",
        "pred_edge_dist", "pred_touches_border", "pred_bbox_missing"]
FORMATF = ["fenced_json", "has_bbox_2d_key", "n_boxes", "raw_len", "parse_success"]
TEXTF = ["expression_length", "has_spatial_words", "has_relation_words", "has_attribute_words"]
OUTCOME = "click_fail"


def heads_def(tokmiss):
    return {
        "geom_logistic": (GEOM, "log"),
        "geom_gbdt": (GEOM, "gbdt"),
        "full_logistic": (GEOM + FORMATF + TEXTF, "log"),
        "full_gbdt": (GEOM + FORMATF + TEXTF, "gbdt"),
        "token_logistic": (TOKENF + tokmiss, "log"),
        "token_gbdt": (TOKENF + tokmiss, "gbdt"),
        "geom_token_logistic": (GEOM + TOKENF + tokmiss, "log"),
        "geom_token_gbdt": (GEOM + TOKENF + tokmiss, "gbdt"),
    }


def main():
    w = pd.read_parquet(REPO / "www2027_revision/outputs/e3_web/per_row_web.parquet")
    models = sorted(w["model"].unique())
    rows_metrics, mc_rows, split_rows, site_rows, subset_rows = [], [], [], [], []
    scores_store = {}
    for m in models:
        dm = w[w["model"] == m]
        fit = dm[dm["split"] == "fit"].copy()
        calib = dm[dm["split"] == "calib"].copy()
        tests = {s: dm[dm["split"] == s].copy() for s in ["test_website", "test_domain"]}
        med = {k: float(np.nanmedian(fit[k].astype(float))) for k in TOKENF}
        allf = {"fit": fit, "calib": calib, **tests}
        for df in allf.values():
            for k in TOKENF:
                df[k + "_missing"] = df[k].isna().astype(float)
                df[k] = df[k].astype(float).fillna(med[k] if np.isfinite(med[k]) else 0.0)
        tokmiss = [k + "_missing" for k in TOKENF]
        HD = heads_def(tokmiss)
        ytr = fit[OUTCOME].to_numpy()
        preds = {s: {} for s in ["calib", "test_website", "test_domain"]}
        for head, (cols, kind) in HD.items():
            Xtr = fit[cols].to_numpy(float)
            for s in preds:
                Xte = allf[s][cols].to_numpy(float)
                if kind == "log":
                    preds[s][head] = E.fit_logistic(Xtr, ytr == 0, Xte, SEED_FIT)
                else:
                    preds[s][head] = E.fit_gbdt_nested(Xtr, ytr == 0, fit["task"].to_numpy(), Xte, SEED_FIT)
        # raw entropy reference
        for s in preds:
            v = allf[s]["coord_mean_entropy"].to_numpy(float)
            preds[s]["raw_coord_mean_entropy"] = v  # FAILURE score on a common raw scale; not a probability
        # store per-row scores
        for s in ["calib", "test_website", "test_domain"]:
            df = allf[s][["sample_id", "task", "website", "domain", "op", OUTCOME, "iou_fail", "gt_area"]].copy()
            df["split"] = s
            df["accepted_frozen_raw_entropy"] = np.asarray(preds[s]["raw_coord_mean_entropy"]) <= np.median(preds["calib"]["raw_coord_mean_entropy"])
            for head, p in preds[s].items():
                df["raw_entropy_failure_score" if head == "raw_coord_mean_entropy" else "prob_success_" + head] = p
            scores_store.setdefault(m, []).append(df)
        # point metrics
        for s in ["test_website", "test_domain"]:
            df = allf[s]
            y = df[OUTCOME].to_numpy()
            tie = np.random.default_rng(SEED_FIT + 7).uniform(size=len(df))
            for head, p in preds[s].items():
                risk_score = np.asarray(p,float) if head == "raw_coord_mean_entropy" else 1-np.asarray(p,float)
                pm = E.point_metrics(1-risk_score, y, tie)  # rank metrics only
                k20 = int(np.ceil(0.2 * len(df)))
                order = np.lexsort((tie, -risk_score))
                caught = int(y[order[:k20]].sum())
                rows_metrics.append(dict(model=m, split=s, method=head, n=len(df),
                                         n_fail=int(y.sum()), fit_inputs=len(fit), fit_labels=0 if head == "raw_coord_mean_entropy" else len(fit),
                                         threshold_inputs=len(calib), threshold_labels=0,
                                         caught_at_20=caught, k20=k20, **pm))
            # MC random baseline
            rng = np.random.default_rng(SEED_MC)
            n = len(df); k20 = int(np.ceil(0.2 * n)); k50 = int(np.ceil(0.5 * n))
            aurocs, aurcs, r50s, c20s = [], [], [], []
            yv = y.astype(float)
            for b in range(NMC):
                perm = rng.permutation(n)
                yp = yv[perm]
                cum = np.cumsum(yp); kk = np.arange(1, n + 1)
                risk = cum / kk
                aurcs.append(float(risk.mean()))
                r50s.append(float(risk[k50 - 1]))
                c20s.append(float(cum[k20 - 1]))
                # AUROC of a random ranking: rank-sum formula on permuted order
                s_fail = np.empty(n); s_fail[perm] = np.arange(n)  # random scores
                aurocs.append(E._rank_auc(s_fail, y))
            prev = float(y.mean())
            mc_rows.append(dict(model=m, split=s, n=n, n_fail=int(y.sum()), k20=k20,
                                mc_auroc_mean=float(np.mean(aurocs)), mc_auroc_lo=float(np.quantile(aurocs, .025)), mc_auroc_hi=float(np.quantile(aurocs, .975)),
                                mc_aurc_mean=float(np.mean(aurcs)),
                                mc_r50_mean=float(np.mean(r50s)), mc_r50_lo=float(np.quantile(r50s, .025)), mc_r50_hi=float(np.quantile(r50s, .975)),
                                mc_c20_mean=float(np.mean(c20s)), mc_c20_lo=float(np.quantile(c20s, .025)), mc_c20_hi=float(np.quantile(c20s, .975)),
                                theory_c20=k20 * int(y.sum()) / n, theory_r=prev, theory_auroc=0.5))
        # frozen thresholds (calib) applied to both tests for key heads
        for head in ["geom_token_logistic", "geom_gbdt", "token_logistic", "geom_token_gbdt", "raw_coord_mean_entropy"]:
            pc = np.asarray(preds["calib"][head], float)
            for cov in ([0.5] if head == "raw_coord_mean_entropy" else [0.5, 0.7, 0.8, 0.9]):
                calrisk = pc if head == "raw_coord_mean_entropy" else 1-pc
                thr = np.quantile(calrisk, cov)
                for s in ["test_website", "test_domain"]:
                    pt = np.asarray(preds[s][head], float)
                    y = allf[s][OUTCOME].to_numpy()
                    testrisk = pt if head == "raw_coord_mean_entropy" else 1-pt
                    acc = testrisk <= thr
                    rows_metrics.append(dict(model=m, split=s, method=f"{head}@thr{int(cov*100)}", n=len(y),
                                             n_fail=int(y.sum()), fit_inputs=len(fit), fit_labels=0 if head == "raw_coord_mean_entropy" else len(fit),
                                         threshold_inputs=len(calib), threshold_labels=0,
                                             caught_at_20=None, k20=None, auroc=None, auprc=None,
                                             prevalence=float(y.mean()), aurc=None,
                                             risk_at_50=None, risk_at_70=None, risk_at_80=None, risk_at_90=None,
                                             thr_cov=float(acc.mean()), thr_risk=float(y[acc].mean()) if acc.any() else None))
        print(m, "heads+MC done", flush=True)

    pd.DataFrame(rows_metrics).to_csv(OUT / "WEB_STRONG_BASELINES.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(OUT / "WEB_RANDOM_BASELINE_MC.csv", index=False)
    for m, parts in scores_store.items():
        pd.concat(parts, ignore_index=True).to_parquet(OUT / f"per_row_scores_{m}.parquet", index=False)

    # paired synchronized bootstrap deltas on test_website
    specs = [("geom_token_logistic", "geom_gbdt"), ("geom_token_logistic", "token_logistic"),
             ("geom_token_gbdt", "geom_gbdt"), ("geom_token_gbdt", "token_gbdt"),
             ("token_gbdt", "geom_gbdt"), ("geom_token_gbdt", "geom_token_logistic")]
    stores = {}
    for m in models:
        df = pd.concat(scores_store[m], ignore_index=True)
        df = df[df["split"] == "test_website"].reset_index(drop=True)
        stores[m] = df
    tasks = sorted(set.union(*[set(stores[m]["task"]) for m in models]))
    ti = {t: i for i, t in enumerate(tasks)}
    idx_by = {m: [np.array([], int)] * len(tasks) for m in models}
    for m in models:
        lists = [[] for _ in tasks]
        for i, t in enumerate(stores[m]["task"]):
            lists[ti[t]].append(i)
        idx_by[m] = [np.array(v, int) for v in lists]
    rng = np.random.default_rng(SEED_FIT)
    draws = {f"{a}-{c}": [] for a, c in specs}
    for b in range(NBOOT):
        dr = rng.integers(0, len(tasks), len(tasks))
        for a, c in specs:
            vals = []
            for m in models:
                df = stores[m]
                idx = np.concatenate([idx_by[m][g] for g in dr if len(idx_by[m][g])])
                y = df[OUTCOME].to_numpy()[idx]
                pa = df["prob_success_" + a].to_numpy(float)[idx]
                pc = df["prob_success_" + c].to_numpy(float)[idx]
                vals.append(E._rank_auc(1 - pa, y) - E._rank_auc(1 - pc, y))
            draws[f"{a}-{c}"].append(float(np.mean(vals)))
    deltas = {k: dict(mean=float(np.mean(v)), lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975)))
              for k, v in draws.items()}
    (OUT / "WEB_DELTAS_BOOTSTRAP.json").write_text(json.dumps(deltas, indent=2))

    # split comparison table
    pm = pd.DataFrame(rows_metrics)
    key_methods = ["geom_gbdt", "token_logistic", "geom_token_logistic", "geom_token_gbdt", "raw_coord_mean_entropy"]
    for m in models:
        for s in ["test_website", "test_domain"]:
            base = pm[(pm["model"] == m) & (pm["split"] == s) & (pm["method"] == "geom_token_logistic")].iloc[0]
            mc = [r for r in mc_rows if r["model"] == m and r["split"] == s][0]
            row = dict(model=m, split=s, n=int(base["n"]), n_fail=int(base["n_fail"]),
                       websites=int(w[(w["model"] == m) & (w["split"] == s)]["website"].nunique()),
                       tasks=int(w[(w["model"] == m) & (w["split"] == s)]["task"].nunique()),
                       prevalence=round(float(base["n_fail"]) / base["n"], 4),
                       random_c20_E=round(mc["theory_c20"], 1))
            for meth in key_methods:
                r = pm[(pm["model"] == m) & (pm["split"] == s) & (pm["method"] == meth)].iloc[0]
                row[f"{meth}_auroc"] = round(float(r["auroc"]), 3)
                row[f"{meth}_c20"] = int(r["caught_at_20"])
                row[f"{meth}_r50"] = round(float(r["risk_at_50"]), 3)
            split_rows.append(row)
    pd.DataFrame(split_rows).to_csv(OUT / "WEB_SPLIT_COMPARISON.csv", index=False)

    # per-website + LOO (test_website, eval-side only)
    for m in models:
        df = stores[m]
        y_all = df[OUTCOME].to_numpy()
        for head in ["token_logistic", "geom_token_logistic"]:
            p_all = df["prob_success_" + head].to_numpy(float)
            tie = np.random.default_rng(SEED_FIT + 7).uniform(size=len(df))
            for site, sub_idx in df.groupby("website").groups.items():
                idx = np.array(sorted(sub_idx))
                y = y_all[idx]
                rowd = dict(model=m, head=head, website=site, n=len(idx),
                            tasks=int(df.loc[idx, "task"].nunique()), prevalence=round(float(y.mean()), 3))
                rowd["auroc"] = round(E._rank_auc(1 - p_all[idx], y), 3) if 0 < y.sum() < len(y) else None
                site_rows.append(rowd)
            # LOO aggregates
            loo = []
            for site in df["website"].unique():
                keep = (df["website"] != site).to_numpy()
                y = y_all[keep]
                loo.append(E._rank_auc(1 - p_all[keep], y))
            site_rows.append(dict(model=m, head=head, website="__LOO_AUROC__", n=len(df["website"].unique()),
                                  tasks=None, prevalence=None,
                                  auroc=f"min {min(loo):.3f} / med {np.median(loo):.3f} / max {max(loo):.3f}"))
    pd.DataFrame(site_rows).to_csv(OUT / "WEB_SITE_SENSITIVITY.csv", index=False)

    # eligibility audit per split
    ex = pd.read_csv(REPO / "www2027_revision/outputs/mind2web_frozen/exclusions.csv")
    idx_all = {s: pd.read_csv(REPO / f"www2027_revision/outputs/mind2web_index/{s}.csv") for s in ["train", "test_website", "test_domain"]}
    el_rows = []
    for s, dfi in idx_all.items():
        exs = ex[ex["split"] == s]
        atoms = pd.Series([r for rs in exs["exclude_reasons"] for r in str(rs).split(";") if r]).value_counts()
        el_rows.append(dict(split=s, official_actions=len(dfi), excluded_actions=len(exs),
                            eligible=len(dfi) - len(exs),
                            **{f"reason_{k}": int(v) for k, v in atoms.items()}))
    eldf = pd.DataFrame(el_rows)
    eldf.to_csv(OUT / "WEB_ELIGIBILITY_AUDIT.csv", index=False)

    # GT-in-crop + op distribution on evaluated test_website population
    man = pd.read_csv(REPO / "www2027_revision/outputs/mind2web_frozen/manifest.csv")
    tw_man = man[man["role_split"] == "test_website"].copy()
    tw_man["rect_bottom"] = tw_man["rect_y"] + tw_man["rect_h"]
    frac_full_in = float((tw_man["rect_bottom"] <= 4096).mean())
    op_dist = tw_man["op"].value_counts().to_dict()

    # subset sensitivity (exploratory, frozen defs)
    for m in models:
        df = stores[m].copy()
        mm = tw_man.set_index(tw_man["annotation_id"].astype(str) + "_" + tw_man["action_uid"].astype(str))
        df["rect_bottom_ok"] = [bool(mm.loc[sid, "rect_bottom"] <= 4096) if sid in mm.index else True for sid in df["sample_id"]]
        subs = {"S1_click_only": df["op"] == "CLICK",
                "S2_gt_full_in_crop": df["rect_bottom_ok"],
                "S3_label_len_ge4": df["sample_id"].map(lambda s: True)}  # label length needs records; approx below
        # S3 from frozen records
        import json as _j
        lab = {}
        for line in open((Path(os.environ.get("VLMGF_REPO", ".")) / "data/web/mind2web_frozen/records/test_website.jsonl")):
            r = _j.loads(line)
            lab[r["sample_id"]] = len(str(r["referring_expression"]).replace('"', '').strip())
        subs["S3_label_len_ge4"] = df["sample_id"].map(lambda s: lab.get(s, 0) >= 10)
        for name, mask in subs.items():
            sub = df[mask.to_numpy()]
            y = sub[OUTCOME].to_numpy()
            if len(sub) < 50 or y.sum() in (0, len(y)):
                subset_rows.append(dict(model=m, subset=name, n=len(sub), note="NA"))
                continue
            for head in ["token_logistic", "geom_token_logistic", "geom_gbdt"]:
                p = sub["prob_success_" + head].to_numpy(float)
                subset_rows.append(dict(model=m, subset=name, n=len(sub), prevalence=round(float(y.mean()), 3),
                                        head=head, auroc=round(E._rank_auc(1 - p, y), 3)))
    pd.DataFrame(subset_rows).to_csv(OUT / "WEB_SUBSET_SENSITIVITY.csv", index=False)

    consistency = dict(frac_gt_fully_in_crop_test_website=round(frac_full_in, 4),
                       op_distribution_test_website=op_dist,
                       deltas=deltas,
                       mc_random={f"{r['model']}|{r['split']}": dict(c20_mean=round(r["mc_c20_mean"], 1),
                                                                    c20_lo=r["mc_c20_lo"], c20_hi=r["mc_c20_hi"],
                                                                    theory_c20=round(r["theory_c20"], 1),
                                                                    auroc_mean=round(r["mc_auroc_mean"], 4)) for r in mc_rows})
    (OUT / "WEB_NUMERIC_CONSISTENCY.json").write_text(json.dumps(consistency, indent=2))
    print(json.dumps(consistency, indent=2)[:2200])


if __name__ == "__main__":
    main()
