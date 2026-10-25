#!/usr/bin/env python
"""Build the deterministic, test-time-proxy-stratified FAR-Ground M2 pilot set."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


QUOTAS = {
    "relation_or_spatial": 40,
    "part_reference": 30,
    "small_or_medium_predicted": 35,
    "high_interface_disagreement": 35,
    "low_router_reliability": 30,
    "easy_proxy_control": 30,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1_dir", default="outputs/method/far_ground/m1_headroom")
    parser.add_argument(
        "--source_predictions",
        default="outputs/predictions/20260713_qwen3vl4b_refcoco_val_1000_qwen_native_box_w1_refcoco1000_native_codex.jsonl",
    )
    parser.add_argument("--output", default="data/processed/far_ground_m2_refcoco_val_200.jsonl")
    parser.add_argument("--manifest", default="outputs/method/far_ground/m2_operator_pilot/sample_manifest.csv")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    m1_dir = Path(args.m1_dir)
    candidates = pd.read_csv(m1_dir / "candidates.csv")
    predictions = pd.read_csv(m1_dir / "strategy_predictions.csv")

    qwen = candidates[candidates.model_key == "qwen3vl4b"].copy()
    intern = candidates[candidates.model_key == "internvl35_4b"].copy()
    qwen_sample = qwen.groupby("sample_id").agg(
        expression_length=("expression_length_log", "first"),
        has_spatial_words=("has_spatial_words", "first"),
        has_relation_words=("has_relation_words", "first"),
        has_attribute_words=("has_attribute_words", "first"),
        likely_part_reference=("likely_part_reference", "first"),
        qwen_agreement=("mean_pair_iou", "mean"),
    )
    qwen_tuple = qwen[qwen.interface == "tuple"].set_index("sample_id")
    qwen_sample["predicted_area"] = qwen_tuple["pred_area_ratio"]
    intern_sample = intern.groupby("sample_id").agg(intern_agreement=("mean_pair_iou", "mean"))
    frame = qwen_sample.join(intern_sample, how="inner")
    frame["agreement"] = frame[["qwen_agreement", "intern_agreement"]].mean(axis=1)

    score_cols = ["full_score_native", "full_score_norm_json", "full_score_tuple"]
    reliability = predictions.groupby("sample_id")[score_cols].max().mean(axis=1)
    frame["router_reliability"] = reliability
    disagreement_cut = float(frame.agreement.quantile(0.40))
    reliable_cut = float(frame.router_reliability.quantile(0.60))
    unreliable_cut = float(frame.router_reliability.quantile(0.40))

    eligibility = {
        "relation_or_spatial": (frame.has_relation_words > 0) | (frame.has_spatial_words > 0),
        "part_reference": frame.likely_part_reference > 0,
        "small_or_medium_predicted": frame.predicted_area < 0.15,
        "high_interface_disagreement": frame.agreement <= disagreement_cut,
        "low_router_reliability": frame.router_reliability <= unreliable_cut,
        "easy_proxy_control": (
            (frame.has_relation_words == 0)
            & (frame.has_spatial_words == 0)
            & (frame.likely_part_reference == 0)
            & (frame.agreement > disagreement_cut)
            & (frame.router_reliability >= reliable_cut)
        ),
    }

    selected = []
    assigned = {}
    for stratum, quota in QUOTAS.items():
        pool = [sample_id for sample_id in frame.index[eligibility[stratum]] if sample_id not in assigned]
        rng.shuffle(pool)
        if len(pool) < quota:
            raise RuntimeError(f"Stratum {stratum} has only {len(pool)} unselected candidates for quota {quota}")
        chosen = pool[:quota]
        selected.extend(chosen)
        assigned.update({sample_id: stratum for sample_id in chosen})
    if len(selected) != 200 or len(set(selected)) != 200:
        raise RuntimeError(f"Expected 200 unique samples, found {len(selected)} / {len(set(selected))}")

    source_records = {str(row["sample_id"]): row["sample"] for row in read_jsonl(args.source_predictions)}
    output_rows = []
    manifest_rows = []
    for position, sample_id in enumerate(selected):
        sample_id = str(sample_id)
        sample = dict(source_records[sample_id])
        metadata = dict(sample.get("metadata") or {})
        flags = {name: bool(eligibility[name].loc[int(sample_id)] if isinstance(frame.index[0], (int, np.integer)) else eligibility[name].loc[sample_id]) for name in QUOTAS}
        metadata["far_ground_m2"] = {
            "selection_position": position,
            "assigned_stratum": assigned[int(sample_id)] if isinstance(next(iter(assigned)), (int, np.integer)) else assigned[sample_id],
            "eligibility": flags,
            "predicted_area_proxy": float(frame.loc[int(sample_id) if isinstance(frame.index[0], (int, np.integer)) else sample_id, "predicted_area"]),
            "cross_interface_agreement_proxy": float(frame.loc[int(sample_id) if isinstance(frame.index[0], (int, np.integer)) else sample_id, "agreement"]),
            "router_reliability_proxy": float(frame.loc[int(sample_id) if isinstance(frame.index[0], (int, np.integer)) else sample_id, "router_reliability"]),
            "selection_seed": args.seed,
        }
        sample["metadata"] = metadata
        output_rows.append(sample)
        manifest_rows.append(
            {
                "selection_position": position,
                "sample_id": sample_id,
                "assigned_stratum": metadata["far_ground_m2"]["assigned_stratum"],
                "expression": sample.get("referring_expression"),
                "predicted_area_proxy": metadata["far_ground_m2"]["predicted_area_proxy"],
                "cross_interface_agreement_proxy": metadata["far_ground_m2"]["cross_interface_agreement_proxy"],
                "router_reliability_proxy": metadata["far_ground_m2"]["router_reliability_proxy"],
                **{f"eligible_{name}": value for name, value in flags.items()},
            }
        )

    write_jsonl(args.output, output_rows)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    counts = Counter(row["assigned_stratum"] for row in manifest_rows)
    print(json.dumps({"output": args.output, "n": len(output_rows), "assigned_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
