#!/usr/bin/env python3
"""R9.1 evaluation (CPU). Two stages so that stage 2 runs from the anonymous artifact.

  parse : raw sampled outputs -> per-sample parsed boxes (same parser v2 rule as the base
          greedy outputs: coordinate_space=norm1000, no decimal recovery, point outputs = failure)
  score : parsed boxes -> SafeGround-derived scores -> metrics on the frozen test prefix,
          paired task-cluster bootstrap, calibration-frozen thresholds, verdict

Pre-specified (EXPERIMENT_PROTOCOL R9.1): primary = AUROC(code_default combined) minus
AUROC(token-only logistic) on the 500 test inputs (click-location failure); 2,000 paired
task-cluster draws, seed 20261009; verdict SAMPLING_BETTER / SINGLE_OUTPUT_BETTER /
INCONCLUSIVE_DIFFERENCE from the 95% percentile interval; non-inferiority NOT_TESTED.

Usage:
  r9_evaluate.py parse --samples <jsonl> --inputs <csv> --out <parsed.jsonl>
  r9_evaluate.py score --parsed <parsed.jsonl> --inputs <csv> --scores <per_row_scores.parquet>
                       --out_dir <dir>
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
SEED = 20261009
NBOOT = 2000
K = 10
SINGLE_OUTPUT = {  # method -> per-row column (probability of success; failure score = 1 - p)
    "token_only": "prob_success_token_logistic",          # PRIMARY comparator (R3 token-only head)
    "raw_coord_entropy": "raw_entropy_failure_score",
    "geom_token": "prob_success_geom_token_logistic",
    "geom_gbdt": "prob_success_geom_gbdt",
    "full_logistic": "prob_success_full_logistic",
}
SAMPLING = {"sg_code_default": "code_default_combined", "sg_paper_params": "paper_params_combined"}
PRIMARY = ("sg_code_default", "token_only")
FIT_LABELS = {"token_only": 510, "raw_coord_entropy": 0, "geom_token": 510, "geom_gbdt": 510,
              "full_logistic": 510, "sg_code_default": 0, "sg_paper_params": 0}


# ---------------------------------------------------------------- stage 1
def cmd_parse(a):
    repo_layout = HERE.parents[2] / "www2027_revision" / "src_rev"      # full repository
    art_layout = HERE.parents[1] / "src" / "parsing_rev"                 # anonymous artifact
    for root, parser_dir in [(HERE.parents[2], repo_layout), (HERE.parents[1], art_layout)]:
        if (parser_dir / "bbox_parser_v2.py").exists():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(parser_dir))
            break
    import bbox_parser_v2 as v2
    point_sources = {"json_point", "point_recovered", "molmo_point"}
    inputs = {r["sample_id"]: r for r in csv.DictReader(open(a.inputs))}
    last = {}
    for line in open(a.samples):  # the last record per input wins (one infrastructure retry allowed)
        rec = json.loads(line)
        last[rec["sample_id"]] = rec
    n = 0
    with open(a.out, "w") as fh:
        for rec in sorted(last.values(), key=lambda r: r["position"]):
            inp = inputs[rec["sample_id"]]
            W, H = int(inp["image_width"]), int(inp["image_height"])
            boxes, sources = [], []
            raws = rec["raw_samples"] if rec.get("error") is None else []
            for raw in raws:
                p = v2.parse_bbox_output(raw or "", W, H, coordinate_space="norm1000",
                                         recover_dropped_decimal=a.recover_dropped_decimal)
                ok = bool(p.get("parse_success")) and p.get("parse_source") not in point_sources
                boxes.append(p["parsed_bbox_xyxy"] if ok else None)
                sources.append(p.get("parse_source") if ok else (p.get("parse_error_type") or "point_output"))
            fh.write(json.dumps(dict(sample_id=rec["sample_id"], position=rec["position"], split=rec["split"],
                                     error=rec.get("error"), sampling_mode=rec.get("sampling_mode"),
                                     n_attempts=len(raws), boxes=boxes, parse_status=sources)) + "\n")
            n += 1
    print("parsed", n, "inputs ->", a.out)


# ---------------------------------------------------------------- metrics
def rank_auc(score_fail, y):
    r = pd.Series(score_fail).rank(method="average").to_numpy()
    pos = np.asarray(y) == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def risk_curve(score_fail, y, tie):
    order = np.lexsort((tie, np.asarray(score_fail, float)))  # ascending failure score = accept first
    f = np.asarray(y)[order]
    return np.cumsum(f) / np.arange(1, len(f) + 1)


def point_metrics(score_fail, y, tie):
    rc = risk_curve(score_fail, y, tie)
    k20 = int(math.ceil(0.2 * len(y)))
    order = np.lexsort((tie, -np.asarray(score_fail, float)))
    return dict(auroc=rank_auc(score_fail, y), aurc=float(rc.mean()),
                risk_at_50=float(rc[int(math.ceil(0.5 * len(y))) - 1]),
                caught_at_20=int(np.asarray(y)[order[:k20]].sum()), k20=k20)


def frozen_threshold(cal_fail, test_fail, cal_y, test_y):
    """Accept iff failure score <= the calibration median (same rule as the R3 heads)."""
    tau = float(np.quantile(cal_fail, 0.5))
    ca, ta = cal_fail <= tau, test_fail <= tau
    return dict(tau=tau, calib_coverage=float(ca.mean()), calib_accepted_risk=float(cal_y[ca].mean()) if ca.any() else float("nan"),
                test_coverage=float(ta.mean()), test_accepted_risk=float(test_y[ta].mean()) if ta.any() else float("nan"))


def verdict(lo, hi):
    if lo > 0:
        return "SAMPLING_BETTER"
    if hi < 0:
        return "SINGLE_OUTPUT_BETTER"
    return "INCONCLUSIVE_DIFFERENCE"


# ---------------------------------------------------------------- stage 2
def build_frame(parsed_path, inputs_path, scores_path):
    sys.path.insert(0, str(HERE.parents[1] / "r9_1"))
    import r9_score
    inputs = {r["sample_id"]: r for r in csv.DictReader(open(inputs_path))}
    rows = []
    for line in open(parsed_path):
        p = json.loads(line)
        inp = inputs[p["sample_id"]]
        s = r9_score.score_input(p["boxes"], int(inp["image_width"]), int(inp["image_height"]),
                                 int(inp["resized_width"]), int(inp["resized_height"]))
        rows.append(dict(sample_id=p["sample_id"], position=int(inp["position"]), split=inp["split"],
                         task=inp["task"], website=inp["website"], n_attempts=p["n_attempts"],
                         run_error=p["error"], sampling_mode=p["sampling_mode"], **s))
    f = pd.DataFrame(rows)
    sc = pd.read_parquet(scores_path)
    sc = sc[sc["split"].isin(["calib", "test_website"])][["sample_id", "click_fail", "iou_fail"] + list(SINGLE_OUTPUT.values())]
    f = f.merge(sc, on="sample_id", how="left", validate="one_to_one")
    assert f["click_fail"].notna().all(), "label join failed"
    for m, col in SINGLE_OUTPUT.items():
        f[m] = f[col].astype(float) if m == "raw_coord_entropy" else 1.0 - f[col].astype(float)
    for m, col in SAMPLING.items():
        f[m] = f[col].astype(float)
    return f.sort_values("position").reset_index(drop=True)


def task_bootstrap(test, methods, n_boot, seed):
    tasks = sorted(test["task"].unique())
    idx = {t: np.where(test["task"].to_numpy() == t)[0] for t in tasks}
    y = test["click_fail"].to_numpy()
    S = {m: test[m].to_numpy(float) for m in methods}
    rng = np.random.default_rng(seed)
    draws, degenerate = [], 0
    for _ in range(n_boot):
        d = rng.integers(0, len(tasks), size=len(tasks))
        sel = np.concatenate([idx[tasks[i]] for i in d])
        yy = y[sel]
        if yy.min() == yy.max():
            degenerate += 1
            continue
        draws.append({m: rank_auc(S[m][sel], yy) for m in methods})
    return pd.DataFrame(draws), degenerate


def cmd_score(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = build_frame(a.parsed, a.inputs, a.scores)
    f.to_csv(out / "r9_per_input_scores.csv", index=False)
    n_excluded_run_error = int(f["run_error"].notna().sum())  # still failing after the one retry
    f = f[f["run_error"].isna()].reset_index(drop=True)          # excluded for ALL methods alike
    cal, test = f[f["split"] == "calib"], f[f["split"] == "test_website"].reset_index(drop=True)
    methods = list(SAMPLING) + list(SINGLE_OUTPUT)
    tie = np.random.default_rng(SEED + 7).uniform(size=len(test))
    y = test["click_fail"].to_numpy()
    point = {m: point_metrics(test[m].to_numpy(float), y, tie) for m in methods}
    thr = {m: frozen_threshold(cal[m].to_numpy(float), test[m].to_numpy(float), cal["click_fail"].to_numpy(),
                               y) for m in methods}
    boot, degenerate = task_bootstrap(test, methods, a.nboot, SEED)
    deltas = {}
    for m in methods:
        if m == PRIMARY[1]:
            continue
        dd = boot[m] - boot[PRIMARY[1]]
        deltas[f"{m}-minus-{PRIMARY[1]}"] = dict(point=point[m]["auroc"] - point[PRIMARY[1]]["auroc"],
                                                 lo=float(dd.quantile(0.025)), hi=float(dd.quantile(0.975)))
    for m in ["raw_coord_entropy", "geom_token", "geom_gbdt", "full_logistic"]:
        dd = boot[PRIMARY[0]] - boot[m]
        deltas[f"{PRIMARY[0]}-minus-{m}"] = dict(point=point[PRIMARY[0]]["auroc"] - point[m]["auroc"],
                                                 lo=float(dd.quantile(0.025)), hi=float(dd.quantile(0.975)))
    prim = deltas[f"{PRIMARY[0]}-minus-{PRIMARY[1]}"]
    test_valid = test["n_valid"].to_numpy()
    summary = dict(
        n_test=len(test), n_calib=len(cal), n_test_tasks=int(test["task"].nunique()),
        n_test_websites=int(test["website"].nunique()), test_failure_rate=float(y.mean()),
        primary=dict(comparison=f"AUROC({PRIMARY[0]}) - AUROC({PRIMARY[1]})", **prim,
                     difference_verdict=verdict(prim["lo"], prim["hi"]), noninferiority="NOT_TESTED"),
        bootstrap=dict(n_requested=a.nboot, n_effective=int(len(boot)), n_degenerate=int(degenerate), seed=SEED,
                       unit="task (annotation_id), paired across methods"),
        point_metrics=point, frozen_threshold_50pct_calibration=thr, deltas=deltas,
        auroc_ci={m: [float(boot[m].quantile(0.025)), float(boot[m].quantile(0.975))] for m in methods},
        fit_labels_used=FIT_LABELS, calibration_labels_for_thresholds=200,
        sampling_quality=dict(
            samples_attempted=int(f["n_attempts"].sum()), samples_valid=int(f["n_valid"].sum()),
            valid_rate=float(f["n_valid"].sum() / max(f["n_attempts"].sum(), 1)),
            inputs_excluded_after_retry=n_excluded_run_error,
            test_inputs_all_valid=int((test_valid == K).sum()), test_inputs_zero_valid=int((test_valid == 0).sum()),
            sampling_modes=f["sampling_mode"].value_counts(dropna=False).to_dict(),
            test_share_single_region=float((test[SAMPLING[PRIMARY[0]].replace("combined", "concentration")] == 0.1).mean())),
    )
    site = []
    for w, g in test.groupby("website"):
        site.append(dict(website=w, n=len(g), failures=int(g["click_fail"].sum()),
                         auroc_sampling=rank_auc(g[PRIMARY[0]], g["click_fail"]),
                         auroc_token_only=rank_auc(g[PRIMARY[1]], g["click_fail"])))
    pd.DataFrame(site).to_csv(out / "r9_site_descriptive.csv", index=False)
    (out / "r9_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["primary"], indent=2, default=float))
    print("effective bootstrap draws:", len(boot), "degenerate:", degenerate)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--samples", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recover_dropped_decimal", action="store_true",
                   help="the base parse rule of InternVL3.5-4B (False for Qwen3-VL-4B)")
    s = sub.add_parser("score")
    s.add_argument("--parsed", required=True)
    s.add_argument("--inputs", required=True)
    s.add_argument("--scores", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()
    cmd_parse(a) if a.cmd == "parse" else cmd_score(a)


if __name__ == "__main__":
    main()
