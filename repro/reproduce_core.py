#!/usr/bin/env python3
"""Reproduce the paper's quantitative claims from the files shipped in this artifact.

Levels
  L1  recompute metrics and intervals from shipped per-row scores, labels, groups, folds.
  L2  refit score heads and the conditional factor model from shipped features and
      frozen folds, then compare with the shipped scores/coefficients.
  L3  regenerate raw model outputs from images -- NOT supported here (needs model
      checkpoints, GPUs, and the raw datasets; see README "Reproduction levels").

Usage (from the artifact root):
  python repro/reproduce_core.py                 # L1 + L2, full bootstrap draws
  python repro/reproduce_core.py --level L1
  python repro/reproduce_core.py --boot 200      # faster; interval checks then use a looser tolerance
  python repro/reproduce_core.py --skip-gbdt     # L2 without the (slower) GBDT refits
Exit code is non-zero if any check fails.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")  # sklearn GBDT on 8k rows is slower with many threads

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data_compact"
sys.path.insert(0,str(ROOT / "src/scripts_rev"))
from entropy_scale import load_scores, failure_score
SEED = 20261009
RESULTS = []


def check(section, name, got, want, tol=None):
    if tol is None or isinstance(want, (bool, str, list, tuple)):
        ok = got == want
    else:
        ok = bool(np.isfinite(got)) and abs(got - want) <= tol
    RESULTS.append((section, name, got, want, ok))
    g = f"{got:.4f}" if isinstance(got, float) else repr(got)
    w = f"{want:.4f}" if isinstance(want, float) else repr(want)
    print(f"{'PASS' if ok else 'FAIL'}  [{section}] {name}: got {g} expect {w}")


# ---------------------------------------------------------------- metric helpers
def rank_auc(score_fail, y_fail):
    r = pd.Series(score_fail).rank(method="average").to_numpy()
    pos = np.asarray(y_fail) == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def risk_at(prob_success, y_fail, tie, cov):
    order = np.lexsort((tie, -np.asarray(prob_success, float)))
    fail = np.asarray(y_fail)[order]
    risk = np.cumsum(fail) / np.arange(1, len(fail) + 1)
    return float(risk[max(int(np.ceil(cov * len(risk))) - 1, 0)])


def caught_at(prob_success, y_fail, tie, frac):
    k = int(np.ceil(frac * len(y_fail)))
    order = np.lexsort((tie, -(1 - np.asarray(prob_success, float))))
    return int(np.asarray(y_fail)[order[:k]].sum())


def group_index(groups_per_unit, universe):
    gi = {g: i for i, g in enumerate(universe)}
    lists = [[] for _ in universe]
    for i, g in enumerate(groups_per_unit):
        lists[gi[g]].append(i)
    return [np.array(v, int) for v in lists]


def sync_boot_delta(units, a, c, n_boot, seed):
    """units: list of dicts with y, groups, scores{method}; one group multiset per draw for all units."""
    universe = sorted(set().union(*[set(u["groups"]) for u in units]))
    idx = [group_index(u["groups"], universe) for u in units]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(universe), size=len(universe))
        vals = []
        for u, ix in zip(units, idx):
            sel = np.concatenate([ix[g] for g in draw if len(ix[g])])
            y = u["y"][sel]
            vals.append(rank_auc(1 - u["scores"][a][sel], y) - rank_auc(1 - u["scores"][c][sel], y))
        out.append(np.mean(vals))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# ---------------------------------------------------------------- L1 sections
def l1_natural(n_boot, tol_ci):
    S = "L1 natural"
    frames = {f.stem.replace("per_row_", ""): pd.read_parquet(f)
              for f in sorted(D.glob("per_row_*.parquet")) if f.name != "per_row_e2.parquet"}
    primary = [m for m in frames if m not in ("qwen3vl4b", "internvl35_4b")]
    for label, models, exp in [
        ("nine models", list(frames), dict(raw_conf=0.5104, geom_gbdt=0.7031, full_logistic=0.7034, full_gbdt=0.7482)),
        ("seven primary models", primary, dict(raw_conf=0.5134, geom_gbdt=0.7218, full_logistic=0.7168, full_gbdt=0.7658)),
    ]:
        for meth, want in exp.items():
            got = np.mean([rank_auc(1 - frames[m][f"prob_success_{meth}"], frames[m]["y_fail"]) for m in models])
            check(S, f"{label}: macro AUROC {meth}", got, want, 5e-4)
    base = np.mean([frames[m]["y_fail"].mean() for m in frames])
    check(S, "nine models: macro accept-all failure", float(base), 0.2842, 5e-4)
    r50 = np.mean([risk_at(frames[m]["prob_success_full_gbdt"], frames[m]["y_fail"],
                           np.random.default_rng(SEED + 7).uniform(size=len(frames[m])), 0.5) for m in frames])
    check(S, "nine models: macro risk@50 full GBDT", float(r50), 0.1554, 5e-4)
    units = {m: dict(y=frames[m]["y_fail"].to_numpy(), groups=frames[m]["group_id"].astype(str).tolist(),
                     scores={k: frames[m][f"prob_success_{k}"].to_numpy(float)
                             for k in ["geom_gbdt", "full_gbdt"]}) for m in frames}
    t = time.time()
    lo, hi = sync_boot_delta([units[m] for m in frames], "full_gbdt", "geom_gbdt", n_boot, SEED)
    check(S, "nine models: full-geometry GBDT delta CI low", lo, 0.0395, tol_ci)
    check(S, "nine models: full-geometry GBDT delta CI high", hi, 0.0511, tol_ci)
    lo, hi = sync_boot_delta([units[m] for m in primary], "full_gbdt", "geom_gbdt", n_boot, SEED)
    check(S, "seven models: full-geometry GBDT delta CI low", lo, 0.0388, tol_ci)
    check(S, "seven models: full-geometry GBDT delta CI high", hi, 0.0490, tol_ci)
    print(f"      (synchronized group bootstrap, {n_boot} draws: {time.time() - t:.0f}s)")


def l1_web(n_boot, tol_ci):
    S = "L1 web"
    exp = {  # (model, split): {method: (AUROC, risk@50, caught@20)}
        ("qwen3vl4b", "test_website"): dict(geom_token_logistic=(0.710, 0.106, 72), geom_gbdt=(0.616, 0.178, 65),
                                            token_logistic=(0.695, 0.131, 70), full_logistic=(0.615, 0.162, 59),
                                            raw_coord_mean_entropy=(0.671, 0.143, 64)),
        ("internvl35_4b", "test_website"): dict(geom_token_logistic=(0.812, 0.427, 162), geom_gbdt=(0.721, 0.493, 146),
                                                token_logistic=(0.801, 0.441, 163), full_logistic=(0.715, 0.479, 142),
                                                raw_coord_mean_entropy=(0.789, 0.444, 154)),
        ("qwen3vl4b", "test_domain"): dict(geom_token_logistic=(0.751, None, None)),
        ("internvl35_4b", "test_domain"): dict(geom_token_logistic=(0.806, None, None)),
    }
    frozen = {("qwen3vl4b", "test_website"): (0.536, 0.107), ("internvl35_4b", "test_website"): (0.378, 0.348),
              ("qwen3vl4b", "test_domain"): (0.613, 0.074), ("internvl35_4b", "test_domain"): (0.440, 0.293)}
    outcome_def = {"qwen3vl4b": (0.2207, 0.4378, 0.1072, 0.2407), "internvl35_4b": (0.6350, 0.8087, 0.3478, 0.5714)}
    random_c20 = {"qwen3vl4b": 37.7, "internvl35_4b": 108.6}
    stores = {}
    for m in ["qwen3vl4b", "internvl35_4b"]:
        s = load_scores(ROOT, m)
        cal = s[s["split"] == "calib"]
        thr = float(np.quantile(cal["prob_success_geom_token_logistic"], 0.5))
        for split in ["test_website", "test_domain"]:
            t = s[s["split"] == split].reset_index(drop=True)
            y = t["click_fail"].to_numpy()
            tie = np.random.default_rng(SEED + 7).uniform(size=len(t))
            for meth, (a, r, c) in exp[(m, split)].items():
                p = 1 - failure_score(t, meth)  # ranking adapter; entropy is not a probability
                check(S, f"{m} {split} {meth} AUROC", rank_auc(1 - p, y), a, 6e-4)
                if r is not None:
                    check(S, f"{m} {split} {meth} risk@50", risk_at(p, y, tie, 0.5), r, 6e-4)
                    check(S, f"{m} {split} {meth} caught@20%", caught_at(p, y, tie, 0.2), c, 0)
            acc = t["prob_success_geom_token_logistic"].to_numpy() >= thr
            cov_want, risk_want = frozen[(m, split)]
            check(S, f"{m} {split} frozen 50%-calibration threshold: coverage", float(acc.mean()), cov_want, 6e-4)
            check(S, f"{m} {split} frozen threshold: accepted click failure", float(y[acc].mean()), risk_want, 6e-4)
            if split == "test_website":
                k20 = int(np.ceil(0.2 * len(t)))
                check(S, f"{m} random-order expectation caught@20%", k20 * y.sum() / len(t), random_c20[m], 0.05)
                a_c, a_i, f_c, f_i = outcome_def[m]
                check(S, f"{m} accept-all failure, click outcome", float(y.mean()), a_c, 6e-4)
                check(S, f"{m} accept-all failure, IoU outcome", float(t["iou_fail"].mean()), a_i, 6e-4)
                check(S, f"{m} same accepted set, click outcome", float(y[acc].mean()), f_c, 6e-4)
                check(S, f"{m} same accepted set, IoU outcome", float(t["iou_fail"].to_numpy()[acc].mean()), f_i, 6e-4)
                stores[m] = dict(y=y, groups=t["task"].astype(str).tolist(),
                                 scores={k: t[f"prob_success_{k}"].to_numpy(float)
                                         for k in ["geom_token_logistic", "geom_gbdt", "token_logistic"]})
    t0 = time.time()
    for (a, c), (lo_w, hi_w) in {("geom_token_logistic", "geom_gbdt"): (0.058, 0.127),
                                 ("geom_token_logistic", "token_logistic"): (-0.005, 0.030)}.items():
        lo, hi = sync_boot_delta(list(stores.values()), a, c, n_boot, SEED)
        check(S, f"task-cluster delta {a} - {c}: CI low", lo, lo_w, max(tol_ci, 6e-4))
        check(S, f"task-cluster delta {a} - {c}: CI high", hi, hi_w, max(tol_ci, 6e-4))
    print(f"      (task-cluster bootstrap, {n_boot} draws: {time.time() - t0:.0f}s)")


def l1_budgets():
    S = "L1 budgets"
    d = pd.read_csv(D / "e4_natural_label_budget_curves.csv")
    for head, budget, mean_w, sd_w in [("full_gbdt", 4000, 0.7402, 0.0015), ("full_gbdt", 50, 0.5824, 0.0252)]:
        g = d[(d["head"] == head) & (d["budget"] == budget)]
        macro = g.groupby("seed")["auroc"].mean()
        check(S, f"Fig.4 {head}@{budget}: model-macro mean", float(macro.mean()), mean_w, 5e-5)
        check(S, f"Fig.4 {head}@{budget}: SD over 5 label draws (band)", float(macro.std(ddof=1)), sd_w, 5e-5)
    g = d[(d["head"] == "full_gbdt") & (d["budget"] == 4000)]
    check(S, "Fig.4 full_gbdt@4000: SD across models (not drawn)",
          float(g.groupby("model")["auroc"].mean().std(ddof=1)), 0.0711, 5e-5)
    w = pd.read_csv(D / "e4_web_budget_curves_budgetonly_impute.csv")
    for m, b, want in [("qwen3vl4b", 50, 0.6642), ("qwen3vl4b", 500, 0.6956),
                       ("internvl35_4b", 50, 0.7761), ("internvl35_4b", 500, 0.8013)]:
        x = w[(w["model"] == m) & (w["head"] == "token") & (w["budget"] == b)]["auroc"]
        check(S, f"Web token-only {m}@{b}: mean over 5 draws", float(x.mean()), want, 5e-5)
    x = w[(w["model"] == "qwen3vl4b") & (w["head"] == "token") & (w["budget"] == 50)]["auroc"]
    check(S, "Web token-only qwen3vl4b@50: min draw", float(x.min()), 0.6236, 5e-5)
    check(S, "Web token-only qwen3vl4b@50: max draw", float(x.max()), 0.7063, 5e-5)


def l1_prospective():
    S = "L1 prospective"
    p = pd.read_csv(D / "prospective_grc_per_row.csv")
    y, s = p["y_fail"].to_numpy(), p["prob_success_grc_full"].to_numpy(float)
    check(S, "Fig.1(C) packet size", len(p), 500, 0)
    check(S, "Fig.1(C) accept-all failure", float(y.mean()), 0.142, 5e-4)
    check(S, "Fig.1(C) risk@50 (original linear GRC)", risk_at(s, y, p["tie_rank"].to_numpy(), 0.5), 0.080, 5e-4)
    check(S, "Fig.1(C) AUROC", rank_auc(1 - s, y), 0.6870, 5e-4)


def _l1_sampling(S, base, samples, inputs, model, recover, k10_mean, cap, n_cap):
    """Re-parse shipped raw samples and recompute the shipped summary (R9.1 / R9.2); also the
    per-input K=10 latency (Table 8 'Extra') and the number of samples at the output-length cap."""
    import json as _json
    import subprocess
    import tempfile
    code = ROOT / "r9_1"
    if not (base / "eval" / "r9_summary.json").exists():
        print(f"SKIP  [{S}] no shipped results")
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cmd = [sys.executable, str(code / "r9_evaluate.py"), "parse", "--samples", str(base / samples),
               "--inputs", str(base / inputs), "--out", str(td / "parsed.jsonl")]
        subprocess.run(cmd + (["--recover_dropped_decimal"] if recover else []), check=True, capture_output=True)
        check(S, "re-parsed boxes identical to shipped parsed.jsonl",
              (td / "parsed.jsonl").read_bytes() == (base / "parsed.jsonl").read_bytes(), True)
        subprocess.run([sys.executable, str(code / "r9_evaluate.py"), "score", "--parsed", str(base / "parsed.jsonl"),
                        "--inputs", str(base / inputs),
                        "--scores", str(D / f"corrections/entropy_scale_v1/per_row_scores_{model}.parquet"),
                        "--out_dir", str(td / "eval")], check=True, capture_output=True)
        got = _json.loads((td / "eval" / "r9_summary.json").read_text())
    ref = _json.loads((D / "corrections/entropy_scale_v1" / ("r9_"+model) / "eval/r9_summary.json").read_text())
    for k in ["point", "lo", "hi"]:
        check(S, f"primary delta {k}", float(got["primary"][k]), float(ref["primary"][k]), 1e-12)
    check(S, "primary difference verdict", got["primary"]["difference_verdict"], ref["primary"]["difference_verdict"])
    for m, v in ref["point_metrics"].items():
        check(S, f"{m} AUROC on the 500 inputs", float(got["point_metrics"][m]["auroc"]), float(v["auroc"]), 1e-12)
    for m, v in ref["frozen_threshold_50pct_calibration"].items():
        g = got["frozen_threshold_50pct_calibration"][m]
        check(S, f"{m} frozen-threshold accepted risk", float(g["test_accepted_risk"]), float(v["test_accepted_risk"]), 1e-12)
        check(S, f"{m} frozen-threshold calib/test coverage (rule: score <= calib median)",
              (g["calib_coverage"], g["test_coverage"]), (v["calib_coverage"], v["test_coverage"]))
    import csv as _csv
    ids = [r["sample_id"] for r in _csv.DictReader(open(base / inputs))]
    last = {}
    for line in open(base / samples):
        r = _json.loads(line)
        last[r["sample_id"]] = r
    sec = sum(last[i]["gen_seconds"] for i in ids) / len(ids)
    check(S, "mean seconds of one batched K=10 call over the 700 inputs (Table 8 'Extra')", round(sec, 7), k10_mean, 1e-12)
    check(S, f"samples reaching the {cap}-token output cap", sum(t >= cap for i in ids for t in last[i]["n_generated_tokens"]), n_cap)



def l1_r9_1():
    _l1_sampling("L1 R9.1 Qwen", ROOT / "r9_1", "qwen3vl4b_r9_1_samples.jsonl", "inputs_frozen.csv", "qwen3vl4b", False,
                 8.2637145, 128, 25)


def l1_r9_2():
    _l1_sampling("L1 R9.2 InternVL", ROOT / "r9_1" / "internvl35_4b", "internvl35_4b_r9_2_samples.jsonl",
                 "inputs_frozen_internvl35_4b.csv", "internvl35_4b", True, 3.1451124, 160, 0)

# ---------------------------------------------------------------- L2 sections
def fit_logistic(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


GBDT_GRID = [(2, 100), (2, 200), (3, 100), (3, 200)]


def fit_gbdt_nested(Xtr, ytr, gtr, Xte):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold
    best, best_loss = None, np.inf
    for depth, iters in GBDT_GRID:
        losses = []
        for tr, va in GroupKFold(n_splits=3).split(Xtr, ytr, gtr):
            if len(np.unique(ytr[tr])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_depth=depth, max_iter=iters, learning_rate=0.1, random_state=SEED)
            m.fit(Xtr[tr], ytr[tr])
            losses.append(log_loss(ytr[va], np.clip(m.predict_proba(Xtr[va])[:, 1], 1e-6, 1 - 1e-6),
                                   labels=[False, True]))
        loss = float(np.mean(losses)) if losses else np.inf
        if loss < best_loss:
            best_loss, best = loss, (depth, iters)
    m = HistGradientBoostingClassifier(max_depth=best[0], max_iter=best[1], learning_rate=0.1, random_state=SEED)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


GEOM = ["pred_area_ratio", "pred_cx", "pred_cy", "pred_w", "pred_h", "pred_aspect",
        "pred_edge_dist", "pred_touches_border", "pred_bbox_missing"]
NAT_FULL = ["confidence", "confidence_missing", "expression_length", "has_spatial_words",
            "has_relation_words", "has_attribute_words", "likely_part_reference"] + GEOM + \
           ["parse_source_json", "parse_success", "fenced_json", "has_bbox_2d_key", "n_boxes", "raw_len"]


def l2_natural(skip_gbdt):
    S = "L2 natural"
    f = pd.read_parquet(D / "natural_features_9model.parquet")
    heads = [("geom_logistic", GEOM, "log"), ("full_logistic", NAT_FULL, "log")]
    if not skip_gbdt:
        heads += [("geom_gbdt", GEOM, "gbdt"), ("full_gbdt", NAT_FULL, "gbdt")]
    aucs = {h[0]: [] for h in heads}
    maxdiff = {h[0]: 0.0 for h in heads}
    for m, g in f.groupby("model", sort=True):
        shipped = pd.read_parquet(D / f"per_row_{m}.parquet")
        y = g["y_fail"].to_numpy()
        folds, groups = g["fold"].to_numpy(), g["group_id"].astype(str).to_numpy()
        for name, cols, kind in heads:
            X = g[cols].to_numpy(float)
            p = np.full(len(g), np.nan)
            for k in sorted(set(folds)):
                tr, te = folds != k, folds == k
                p[te] = (fit_logistic(X[tr], y[tr] == 0, X[te]) if kind == "log"
                         else fit_gbdt_nested(X[tr], y[tr] == 0, groups[tr], X[te]))
            aucs[name].append(rank_auc(1 - p, y))
            maxdiff[name] = max(maxdiff[name], float(np.max(np.abs(p - shipped[f"prob_success_{name}"].to_numpy()))))
    want = dict(geom_logistic=0.6759, full_logistic=0.7034, geom_gbdt=0.7031, full_gbdt=0.7482)
    for name in aucs:
        check(S, f"refit nine-model macro AUROC {name}", float(np.mean(aucs[name])), want[name], 5e-4)
        check(S, f"refit vs shipped per-row scores {name}: max |diff| < 1e-6", maxdiff[name] < 1e-6, True)


def l2_web(skip_gbdt):
    S = "L2 web"
    TOKENF = ["coord_mean_logprob", "coord_min_logprob", "coord_mean_entropy", "out_mean_logprob"]
    MISS = [k + "_missing" for k in TOKENF]
    FORMATF = ["fenced_json", "has_bbox_2d_key", "n_boxes", "raw_len", "parse_success"]
    TEXTF = ["expression_length", "has_spatial_words", "has_relation_words", "has_attribute_words"]
    heads = {"geom_logistic": (GEOM, "log"), "full_logistic": (GEOM + FORMATF + TEXTF, "log"),
             "token_logistic": (TOKENF + MISS, "log"), "geom_token_logistic": (GEOM + TOKENF + MISS, "log")}
    if not skip_gbdt:
        heads.update({"geom_gbdt": (GEOM, "gbdt"), "geom_token_gbdt": (GEOM + TOKENF + MISS, "gbdt")})
    w = pd.read_parquet(D / "e3_web/per_row_web.parquet")
    for m, dm in w.groupby("model"):
        fit = dm[dm["split"] == "fit"].copy()
        test = dm[dm["split"] == "test_website"].copy()
        med = {k: float(np.nanmedian(fit[k].astype(float))) for k in TOKENF}
        for df in (fit, test):
            for k in TOKENF:
                df[k + "_missing"] = df[k].isna().astype(float)
                df[k] = df[k].astype(float).fillna(med[k])
        shipped = pd.read_parquet(D / f"final_audit/per_row_scores_{m}.parquet")
        shipped = shipped[shipped["split"] == "test_website"].reset_index(drop=True)
        ytr = fit["click_fail"].to_numpy()
        for name, (cols, kind) in heads.items():
            Xtr, Xte = fit[cols].to_numpy(float), test[cols].to_numpy(float)
            p = (fit_logistic(Xtr, ytr == 0, Xte) if kind == "log"
                 else fit_gbdt_nested(Xtr, ytr == 0, fit["task"].to_numpy(), Xte))
            diff = float(np.max(np.abs(p - shipped[f"prob_success_{name}"].to_numpy())))
            check(S, f"{m} refit {name} vs shipped test scores: max |diff| < 1e-6", diff < 1e-6, True)


def l2_glm(n_boot):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    S = "L2 GLM"
    frame = pd.read_parquet(D / "joint_factor_input_rev1.parquet")
    valid = frame[frame["parse_valid"] == 1].copy()

    def terms(fr):
        t = ["expression_length_z", "has_spatial", "has_relation", "has_attribute", "likely_part",
             "target_edge", "border_near", "C(model_slug)"]
        for size in ["object_size_medium", "object_size_small"]:
            if fr[size].nunique() > 1:
                t.append(size)
                if fr["likely_part"].nunique() > 1:
                    t.append("likely_part:" + size)
        t.append("C(dataset_family)")
        return [x for x in t if x.startswith("C(") or x not in fr or fr[x].nunique() > 1]

    def form(outcome, fr):
        return outcome + " ~ " + " + ".join(terms(fr))

    check(S, "parse-valid rows", len(valid), 82800, 0)
    prim = smf.glm(form("success", valid), data=valid, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": valid["source_image_id"]}, maxiter=100)
    iou = smf.glm(form("iou_logit", valid), data=valid, family=sm.families.Gaussian()).fit(
        cov_type="cluster", cov_kwds={"groups": valid["source_image_id"]}, maxiter=100)
    shipped = pd.read_csv(D / "joint_coefficients_rev1.csv")
    shipped = shipped[shipped["analysis"] == "rev_valid_success_primary"].set_index("term_raw")["estimate"]
    diff = max(abs(float(prim.params[t]) - float(shipped[t])) for t in prim.params.index)
    check(S, "primary GLM coefficients vs shipped: max |diff| < 1e-6", diff < 1e-6, True)
    # source-image cluster multiplier bootstrap (one-step score perturbation, Rademacher weights)
    exog, endog = np.asarray(prim.model.exog, float), np.asarray(prim.model.endog, float)
    fitted = np.asarray(prim.fittedvalues, float)
    var = np.clip(fitted * (1 - fitted), 1e-7, None)
    bread = np.linalg.pinv(exog.T @ (var[:, None] * exog))
    codes, uniq = pd.factorize(valid["source_image_id"], sort=True)
    scores = np.zeros((len(uniq), exog.shape[1]))
    np.add.at(scores, codes, exog * (endog - fitted)[:, None])
    rng = np.random.default_rng(SEED)
    params = np.asarray(prim.params, float)
    draws = []
    for start in range(0, n_boot, 100):
        size = min(100, n_boot - start)
        wts = rng.choice([-1.0, 1.0], size=(size, len(uniq)))
        draws.append(params[None, :] + (wts @ scores) @ bread.T)
    draws = pd.DataFrame(np.vstack(draws), columns=list(prim.params.index))
    lo_rows = {}
    for col in ["model_slug", "dataset_family"]:
        for held in sorted(valid[col].unique()):
            tr = valid[valid[col] != held]
            res = smf.glm(form("success", tr), data=tr, family=sm.families.Binomial()).fit(maxiter=100)
            lo_rows.setdefault(col, []).append(res.params)
    cats = {"expression_length": ["expression_length_z"], "spatial_cue": ["has_spatial"],
            "relation_cue": ["has_relation"], "attribute_cue": ["has_attribute"],
            "part_reference": ["likely_part"], "target_size": ["object_size_medium", "object_size_small"],
            "target_location": ["target_edge"], "border_proximity": ["border_near"],
            "part_x_size": ["likely_part:object_size_medium", "likely_part:object_size_small"]}
    stable = set()
    for cat, ts in cats.items():
        for t in ts:
            est = float(prim.params[t])
            lo, hi = draws[t].quantile(0.025), draws[t].quantile(0.975)
            iou_ok = np.sign(float(iou.params[t])) == np.sign(est)
            lom = np.mean([np.sign(p[t]) == np.sign(est) for p in lo_rows["model_slug"] if t in p.index])
            lof = np.mean([np.sign(p[t]) == np.sign(est) for p in lo_rows["dataset_family"] if t in p.index])
            if (lo > 0 or hi < 0) and iou_ok and lom >= 0.8 and lof >= 0.8:
                stable.add(cat)
    check(S, "stable factor categories (re-derived gate)", sorted(stable),
          sorted(["attribute_cue", "border_proximity", "expression_length", "part_reference",
                  "part_x_size", "target_location", "target_size"]), None)


def l2_token_panel(skip_gbdt):
    S = "L2 token panel"
    from sklearn.model_selection import GroupKFold
    TOKENF = ["coord_mean_logprob", "coord_min_logprob", "coord_mean_entropy", "out_mean_logprob"]
    MISS = [k + "_missing" for k in TOKENF]
    sets = {"token_only": TOKENF + MISS, "geom": GEOM, "geom_token": GEOM + TOKENF + MISS}
    want = {"qwen3vl4b": dict(token_only=0.664, geom=0.640, geom_token=0.738, raw_entropy=0.627),
            "internvl35_4b": dict(token_only=0.766, geom=0.627, geom_token=0.830, raw_entropy=0.761)}
    for m in want:
        df = pd.read_parquet(D / f"panel_{m}.parquet")
        y, groups = df["y_fail"].to_numpy(), df["group"].to_numpy()
        folds = np.full(len(df), -1)
        order = np.argsort(df["sample_id"].astype(str).to_numpy())
        for k, (_, te) in enumerate(GroupKFold(5).split(df.iloc[order], y[order], groups[order])):
            folds[order[te]] = k
        for name, cols in sets.items():
            X = df[cols].to_numpy(float)
            p = np.full(len(df), np.nan)
            for k in range(5):
                tr, te = folds != k, folds == k
                p[te] = fit_logistic(X[tr], y[tr] == 0, X[te])
            check(S, f"{m} {name} logistic AUROC", rank_auc(1 - p, y), want[m][name], 6e-4)
        check(S, f"{m} raw coordinate entropy AUROC", rank_auc(df["coord_mean_entropy"].to_numpy(float), y),
              want[m]["raw_entropy"], 6e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["L1", "L2", "all"], default="all")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--skip-gbdt", action="store_true")
    ap.add_argument("--only", nargs="*", help='run only these sections, e.g. --only "L2 GLM"')
    a = ap.parse_args()
    tol_ci = 1e-4 if a.boot >= 2000 else 0.01
    timings = []
    sections = []
    if a.level in ("L1", "all"):
        sections += [("L1 natural", lambda: l1_natural(a.boot, tol_ci)), ("L1 web", lambda: l1_web(a.boot, tol_ci)),
                     ("L1 budgets", l1_budgets), ("L1 prospective", l1_prospective), ("L1 R9.1", l1_r9_1), ("L1 R9.2", l1_r9_2)]
    if a.level in ("L2", "all"):
        sections += [("L2 GLM", lambda: l2_glm(a.boot)), ("L2 web", lambda: l2_web(a.skip_gbdt)),
                     ("L2 token panel", lambda: l2_token_panel(a.skip_gbdt)),
                     ("L2 natural", lambda: l2_natural(a.skip_gbdt))]
    if a.only:
        sections = [x for x in sections if x[0] in a.only]
    for name, fn in sections:
        t = time.time()
        fn()
        timings.append((name, time.time() - t))
    n_fail = sum(not r[4] for r in RESULTS)
    print("\nsection timings (s): " + ", ".join(f"{n} {s:.0f}" for n, s in timings))
    print(f"{len(RESULTS)} checks, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
