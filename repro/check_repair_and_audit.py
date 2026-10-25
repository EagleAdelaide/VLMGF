"""Re-derive the prompt-repair gate flags and the manual-review counts from shipped files.

Run from the artifact root:  python repro/check_repair_and_audit.py
Exits non-zero on any mismatch.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PR = ROOT / "prompt_repair"
HA = ROOT / "human_audit"
fails = []


def check(name, got, want, tol=5e-5):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} expect {want!r}")
    if not ok:
        fails.append(name)


# M1: per-sample interface selection
ss = pd.read_csv(PR / "m1_interface_selection/strategy_summary.csv")
gd = pd.read_csv(PR / "m1_interface_selection/gate_decision.csv").set_index("model_key")
expect_m1 = {"qwen3vl4b": (0.0176, 0.016, -0.0017), "internvl35_4b": (0.0191, 0.024, -0.0013)}
for model, (gap_e, gap50_e, router_e) in expect_m1.items():
    s = ss[(ss.model_key == model) & (ss.population == "all_outputs")].set_index("strategy")
    v = ss[(ss.model_key == model) & (ss.population == "all_candidates_valid")].set_index("strategy")
    gap = s.loc["oracle", "mean_iou"] - s.loc["best_fixed_global", "mean_iou"]
    gap50 = s.loc["oracle", "iou_at_0_5"] - s.loc["best_fixed_global", "iou_at_0_5"]
    router = s.loc["full_router", "mean_iou"] - s.loc["best_fixed_global", "mean_iou"]
    vgap = v.loc["oracle", "mean_iou"] - v.loc["best_fixed_global", "mean_iou"]
    vgap50 = v.loc["oracle", "iou_at_0_5"] - v.loc["best_fixed_global", "iou_at_0_5"]
    headroom = gap >= 0.025 or gap50 >= 0.035
    recovery_ok = router / gap >= 0.25
    parse_ind = vgap >= 0.025 or vgap50 >= 0.035
    check(f"M1 {model} oracle gap mIoU", round(gap, 4), gap_e)
    check(f"M1 {model} oracle gap IoU@0.5", round(gap50, 3), gap50_e)
    check(f"M1 {model} full-router delta", round(router, 4), router_e)
    check(f"M1 {model} gate pass (re-derived)", bool(headroom and recovery_ok and parse_ind), False)
    check(f"M1 {model} gate pass (shipped)", bool(gd.loc[model, "model_gate_pass"]), False)

# M2: repair prompts, re-derived from per-sample paired deltas
ps = pd.read_csv(PR / "m2_repair_prompts/paired_operator_samples.csv")
og = pd.read_csv(PR / "m2_repair_prompts/operator_gate.csv").set_index(["model_key", "repair"])
expect_m2 = {
    ("qwen3vl4b", "relation_repair"): (131, -0.0130, (False, True, False)),
    ("qwen3vl4b", "part_repair"): (52, -0.0040, (False, False, False)),
    ("internvl35_4b", "relation_repair"): (131, 0.0038, (False, False, True)),
    ("internvl35_4b", "part_repair"): (52, 0.0237, (True, False, True)),
}
for key, (n_e, d_e, flags_e) in expect_m2.items():
    g = ps[(ps.model_key == key[0]) & (ps.repair == key[1])]
    check(f"M2 {key} rows", len(g), 200)
    t, nt = g[g.target_subset], g[~g.target_subset]
    vb = t[t.baseline_parse & t.repair_parse]
    tp = t.delta_iou.mean() >= 0.03 or t.delta_iou_at_0_5.mean() >= 0.05
    ntp = nt.delta_iou.mean() >= -0.01
    pi = len(vb) >= 0.75 * len(t) and vb.delta_iou.mean() > 0
    check(f"M2 {key} target n", len(t), n_e)
    check(f"M2 {key} target dmIoU", round(t.delta_iou.mean(), 4), d_e)
    check(f"M2 {key} target/non-target/parse flags", (bool(tp), bool(ntp), bool(pi)), flags_e)
    check(f"M2 {key} gate (re-derived)", bool(tp and ntp and pi), False)
    check(f"M2 {key} gate (shipped)", bool(og.loc[key, "operator_gate_pass"]), False)
check("M2 all repair outputs parsed", bool(ps.repair_parse.all()), True)

# Manual review counts and sensitivity
lab = pd.read_csv(HA / "labels_single_annotator.csv")
vc = lab.annotator_1_label.value_counts()
invalid = ["incorrect_gt_box", "duplicate_near_duplicate", "corrupted_image_query",
           "invalid_target_query_conversion"]
check("review rows", len(lab), 300)
check("review missing labels", int(lab.annotator_1_label.isna().sum()), 0)
check("review valid_unambiguous", int(vc.get("valid_unambiguous", 0)), 278)
check("review valid_mildly_ambiguous", int(vc.get("valid_mildly_ambiguous", 0)), 0)
check("review invalid", int(vc.reindex(invalid).fillna(0).sum()), 20)
check("review cannot_determine", int(vc.get("cannot_determine", 0)), 2)
check("review second annotator rows", int(lab.annotator_2_label.notna().sum()), 0)
fs = pd.read_csv(HA / "ingestion/filtered_sample_ids.csv")
check("primary exclusion rows", len(fs), 22)
summ = json.loads((HA / "sensitivity/exclusion_sensitivity_summary.json").read_text())
check("sensitivity leader changes", len(summ["headline_leaders_changed"]), 0)
check("sensitivity max |d rho|", round(summ["max_absolute_rank_correlation_change"], 4), 0.0)
check("sensitivity max |d AUROC|", round(summ["max_absolute_grc_auroc_change"], 4), 0.0041)
check("sensitivity max |d AURC|", round(summ["max_absolute_grc_aurc_change"], 4), 0.0039)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAIL'}")
sys.exit(1 if fails else 0)
