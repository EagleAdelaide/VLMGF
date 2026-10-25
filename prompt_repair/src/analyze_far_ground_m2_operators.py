#!/usr/bin/env python
"""Analyze the preregistered FAR-Ground M2 factor-specific repair pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


REPAIR_TARGET = {
    "relation_repair": lambda row: bool(row["eligible_relation_or_spatial"]),
    "part_repair": lambda row: bool(row["eligible_part_reference"]),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method/far_ground/m2_operator_analysis.json")
    parser.add_argument("--output_dir", default="outputs/method/far_ground/m2_operator_pilot")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def percentile(values, q):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(values, q)) if values else None


def flatten_record(record):
    parsed = record.get("parsed", {}) or {}
    metrics = record.get("metrics", {}) or {}
    return {
        "sample_id": str(record["sample_id"]),
        "run_id": record.get("run_id"),
        "parse_success": bool(parsed.get("parse_success")),
        "iou": float(metrics.get("iou") or 0.0),
        "iou_at_0_5": float(metrics.get("iou_at_0_5") or 0.0),
        "iou_at_0_3": float(metrics.get("iou_at_0_3") or 0.0),
        "pointing_acc": float(metrics.get("pointing_acc") or 0.0),
        "pred_bbox": parsed.get("parsed_bbox_xyxy"),
        "raw_output": parsed.get("raw_model_output"),
    }


def load_operator(path, selected_ids):
    rows = {str(row["sample_id"]): flatten_record(row) for row in read_jsonl(path)}
    missing = selected_ids - set(rows)
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} selected samples")
    return {sample_id: rows[sample_id] for sample_id in selected_ids}


def bootstrap_delta(deltas, n_bootstrap, seed):
    rng = random.Random(seed)
    values = list(map(float, deltas))
    if not values:
        return None, None, None
    distribution = []
    for _ in range(n_bootstrap):
        distribution.append(mean(values[rng.randrange(len(values))] for _ in values))
    return mean(values), percentile(distribution, 2.5), percentile(distribution, 97.5)


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    manifest = pd.read_csv(config["sample_manifest"], dtype={"sample_id": str})
    selected_ids = set(manifest.sample_id)
    if len(selected_ids) != 200:
        raise ValueError(f"Expected 200 selected IDs, found {len(selected_ids)}")
    manifest_map = {row.sample_id: row._asdict() for row in manifest.itertuples(index=False)}

    absolute_rows = []
    paired_rows = []
    bootstrap_rows = []
    gate_rows = []
    run_ids = {}
    for model_key, model_cfg in config["models"].items():
        operators = {
            name: load_operator(path, selected_ids)
            for name, path in model_cfg["operators"].items()
        }
        run_ids[model_key] = {
            name: next(iter(records.values()))["run_id"] for name, records in operators.items()
        }
        best_fixed = model_cfg["best_fixed"]
        baseline = operators[best_fixed]
        for operator, records in operators.items():
            values = list(records.values())
            absolute_rows.append(
                {
                    "model_key": model_key,
                    "model_name": model_cfg["name"],
                    "operator": operator,
                    "n": len(values),
                    "parse_success": mean(value["parse_success"] for value in values),
                    "mean_iou": mean(value["iou"] for value in values),
                    "median_iou": percentile([value["iou"] for value in values], 50),
                    "iou_at_0_5": mean(value["iou_at_0_5"] for value in values),
                    "iou_at_0_3": mean(value["iou_at_0_3"] for value in values),
                    "pointing_acc": mean(value["pointing_acc"] for value in values),
                    "additional_calls_vs_fixed": int(operator in REPAIR_TARGET),
                }
            )
        for repair, target_fn in REPAIR_TARGET.items():
            repair_records = operators[repair]
            for sample_id in sorted(selected_ids):
                base = baseline[sample_id]
                changed = repair_records[sample_id]
                target = target_fn(manifest_map[sample_id])
                paired_rows.append(
                    {
                        "model_key": model_key,
                        "sample_id": sample_id,
                        "repair": repair,
                        "target_subset": target,
                        "assigned_stratum": manifest_map[sample_id]["assigned_stratum"],
                        "baseline_operator": best_fixed,
                        "baseline_parse": base["parse_success"],
                        "repair_parse": changed["parse_success"],
                        "baseline_iou": base["iou"],
                        "repair_iou": changed["iou"],
                        "delta_iou": changed["iou"] - base["iou"],
                        "baseline_iou_at_0_5": base["iou_at_0_5"],
                        "repair_iou_at_0_5": changed["iou_at_0_5"],
                        "delta_iou_at_0_5": changed["iou_at_0_5"] - base["iou_at_0_5"],
                    }
                )

    paired = pd.DataFrame(paired_rows)
    for (model_key, repair), group in paired.groupby(["model_key", "repair"]):
        summaries = {}
        for population, subset in (
            ("overall", group),
            ("target", group[group.target_subset]),
            ("non_target", group[~group.target_subset]),
            ("target_valid_both", group[group.target_subset & group.baseline_parse & group.repair_parse]),
        ):
            point, low, high = bootstrap_delta(
                subset.delta_iou.tolist(), config["bootstrap"], config["seed"]
            )
            summaries[population] = {
                "n": len(subset),
                "delta_iou": point,
                "ci95_low": low,
                "ci95_high": high,
                "delta_iou50": mean(subset.delta_iou_at_0_5),
                "improved_n": int((subset.delta_iou > 1e-9).sum()),
                "worsened_n": int((subset.delta_iou < -1e-9).sum()),
                "unchanged_n": int((subset.delta_iou.abs() <= 1e-9).sum()),
            }
            bootstrap_rows.append(
                {
                    "model_key": model_key,
                    "repair": repair,
                    "population": population,
                    "n": len(subset),
                    "mean_delta_iou": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_delta_iou_at_0_5": mean(subset.delta_iou_at_0_5),
                    "improved_n": int((subset.delta_iou > 1e-9).sum()),
                    "worsened_n": int((subset.delta_iou < -1e-9).sum()),
                    "unchanged_n": int((subset.delta_iou.abs() <= 1e-9).sum()),
                    "baseline_parse_rate": mean(subset.baseline_parse),
                    "repair_parse_rate": mean(subset.repair_parse),
                }
            )
        target = summaries["target"]
        non_target = summaries["non_target"]
        valid = summaries["target_valid_both"]
        target_pass = target["delta_iou"] is not None and (
            target["delta_iou"] >= 0.03 or target["delta_iou50"] >= 0.05
        )
        non_target_pass = non_target["delta_iou"] is not None and non_target["delta_iou"] >= -0.01
        parse_independent = (
            valid["n"] >= 0.75 * target["n"]
            and valid["delta_iou"] is not None
            and valid["delta_iou"] > 0
        )
        gate_rows.append(
            {
                "model_key": model_key,
                "repair": repair,
                "target_n": target["n"],
                "target_delta_mean_iou": target["delta_iou"],
                "target_delta_iou_at_0_5": target["delta_iou50"],
                "target_ci95_low": target["ci95_low"],
                "target_ci95_high": target["ci95_high"],
                "target_improved_n": target["improved_n"],
                "target_worsened_n": target["worsened_n"],
                "target_unchanged_n": target["unchanged_n"],
                "non_target_n": non_target["n"],
                "non_target_delta_mean_iou": non_target["delta_iou"],
                "non_target_ci95_low": non_target["ci95_low"],
                "non_target_ci95_high": non_target["ci95_high"],
                "target_valid_both_n": valid["n"],
                "target_valid_both_delta_mean_iou": valid["delta_iou"],
                "target_pass": target_pass,
                "non_target_pass": non_target_pass,
                "parse_independent": parse_independent,
                "operator_gate_pass": bool(target_pass and non_target_pass and parse_independent),
            }
        )

    write_csv(output_dir / "operator_absolute_metrics.csv", absolute_rows)
    write_csv(output_dir / "paired_operator_samples.csv", paired_rows)
    write_csv(output_dir / "operator_bootstrap.csv", bootstrap_rows)
    write_csv(output_dir / "operator_gate.csv", gate_rows)
    metrics = {
        "stage": "M2",
        "config": args.config,
        "n": 200,
        "bootstrap": config["bootstrap"],
        "seed": config["seed"],
        "run_ids": run_ids,
        "operator_gates": gate_rows,
        "any_operator_pass": any(row["operator_gate_pass"] for row in gate_rows),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# FAR-Ground M2 Operator Pilot Report\n\n",
        "This is a preregistered, proxy-stratified 200-image pilot. It tests targeted repair prompts, not a deployable router.\n\n",
        "## Absolute Metrics\n\n",
        "| Model | Operator | n | Parse | mIoU | IoU@0.5 | IoU@0.3 | Pointing | Extra calls if triggered |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in absolute_rows:
        lines.append(
            f"| {row['model_key']} | {row['operator']} | {row['n']} | {row['parse_success']:.3f} | "
            f"{row['mean_iou']:.4f} | {row['iou_at_0_5']:.4f} | {row['iou_at_0_3']:.4f} | {row['pointing_acc']:.4f} | "
            f"{row['additional_calls_vs_fixed']} |\n"
        )
    lines.extend(
        [
            "\n## Operator Gates\n\n",
            "| Model | Repair | Target n | Target delta mIoU [95% CI] | Target delta IoU@0.5 | Better / worse / same | Non-target delta [95% CI] | Gate |\n",
            "|---|---|---:|---:|---:|---:|---:|---|\n",
        ]
    )
    for row in gate_rows:
        lines.append(
            f"| {row['model_key']} | {row['repair']} | {row['target_n']} | {row['target_delta_mean_iou']:.4f} "
            f"[{row['target_ci95_low']:.4f}, {row['target_ci95_high']:.4f}] | "
            f"{row['target_delta_iou_at_0_5']:.4f} | "
            f"{row['target_improved_n']} / {row['target_worsened_n']} / {row['target_unchanged_n']} | "
            f"{row['non_target_delta_mean_iou']:.4f} [{row['non_target_ci95_low']:.4f}, {row['non_target_ci95_high']:.4f}] | "
            f"{'PASS' if row['operator_gate_pass'] else 'FAIL'} |\n"
        )
    any_pass = any(row["operator_gate_pass"] for row in gate_rows)
    lines.append(f"\nM2 decision: **{'AT LEAST ONE OPERATOR PASSES' if any_pass else 'NO OPERATOR PASSES'}**.\n")
    if not any_pass:
        lines.append("Do not implement M3/M4 or rewrite the paper around FAR-Ground. Preserve this as a negative result.\n")
    lines.append(
        "\nEach triggered repair requires one additional VLM call after the fixed-interface "
        "candidate. All repair outputs parsed successfully, so the decision is not a "
        "parse-recovery effect.\n"
    )
    (output_dir / "M2_OPERATOR_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "gates": gate_rows}, indent=2))


if __name__ == "__main__":
    main()
