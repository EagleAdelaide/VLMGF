#!/usr/bin/env python
"""Preregistered M1 oracle-headroom and grouped routing analysis for FAR-Ground."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_ORDER = ("native", "norm_json", "tuple")
EXPR_FEATURES = (
    "expression_length_log",
    "has_spatial_words",
    "has_relation_words",
    "has_attribute_words",
    "likely_part_reference",
)
CANDIDATE_FEATURES = (
    "parse_success",
    "raw_length_log",
    "confidence",
    "confidence_missing",
    "pred_area_ratio",
    "pred_width_ratio",
    "pred_height_ratio",
    "pred_aspect_log",
    "pred_center_x",
    "pred_center_y",
    "pred_border_proximity",
    "mean_pair_iou",
    "max_pair_iou",
    "mean_center_disagreement",
    "area_disagreement",
    "valid_candidate_count",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--model",
        action="append",
        nargs=7,
        metavar=("MODEL_KEY", "MODEL_NAME", "NATIVE_RUN", "NATIVE_PATH", "JSON_RUN", "JSON_PATH", "TUPLE_PATH"),
        required=True,
        help=(
            "Model key/name, native run/path, normalized-JSON run/path, and tuple path. "
            "Tuple run_id is read from its prediction rows."
        ),
    )
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def percentile(values, q):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(values, q)) if values else None


def box_iou(left, right):
    if left is None or right is None:
        return 0.0
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    larea = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    rarea = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = larea + rarea - inter
    return inter / union if union > 0 else 0.0


def normalized_geometry(box, width, height):
    if box is None or width <= 0 or height <= 0:
        return {
            "pred_area_ratio": 0.0,
            "pred_width_ratio": 0.0,
            "pred_height_ratio": 0.0,
            "pred_aspect_log": 0.0,
            "pred_center_x": 0.5,
            "pred_center_y": 0.5,
            "pred_border_proximity": 0.0,
        }
    x1, y1, x2, y2 = map(float, box)
    bw = max(0.0, x2 - x1) / width
    bh = max(0.0, y2 - y1) / height
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    return {
        "pred_area_ratio": bw * bh,
        "pred_width_ratio": bw,
        "pred_height_ratio": bh,
        "pred_aspect_log": math.log(max(bw, 1e-6) / max(bh, 1e-6)),
        "pred_center_x": cx,
        "pred_center_y": cy,
        "pred_border_proximity": min(cx, 1.0 - cx, cy, 1.0 - cy),
    }


def raw_output_features(record):
    parsed = record.get("parsed", {}) or {}
    raw = str(parsed.get("raw_model_output") or "")
    confidence = parsed.get("confidence")
    return {
        "raw_length_log": math.log1p(len(raw)),
        "confidence": float(confidence) if confidence is not None else 0.5,
        "confidence_missing": 1.0 if confidence is None else 0.0,
    }


def load_matched_model(model_key, model_name, run_specs):
    records_by_interface = {}
    run_ids = {}
    for interface, run_id, path in run_specs:
        records = read_jsonl(path)
        records_by_interface[interface] = {str(row["sample_id"]): row for row in records}
        run_ids[interface] = run_id or (records[0].get("run_id") if records else "")
    common = sorted(set.intersection(*(set(rows) for rows in records_by_interface.values())))
    if len(common) != 1000:
        raise ValueError(f"{model_key}: expected exactly 1000 matched samples, found {len(common)}")

    samples = []
    for sample_id in common:
        source = records_by_interface["native"][sample_id]
        sample = source.get("sample", {}) or {}
        factors = source.get("factors", {}) or {}
        width = float(sample.get("image_width") or 0)
        height = float(sample.get("image_height") or 0)
        candidates = {}
        valid_boxes = []
        for interface in INTERFACE_ORDER:
            record = records_by_interface[interface][sample_id]
            parsed = record.get("parsed", {}) or {}
            parse_success = bool(parsed.get("parse_success"))
            box = parsed.get("parsed_bbox_xyxy") if parse_success else None
            geometry = normalized_geometry(box, width, height)
            candidate = {
                "interface": interface,
                "parse_success": 1.0 if parse_success else 0.0,
                "box": box,
                "iou": float((record.get("metrics", {}) or {}).get("iou") or 0.0),
                "iou_at_0_5": float((record.get("metrics", {}) or {}).get("iou_at_0_5") or 0.0),
                "pointing_acc": float((record.get("metrics", {}) or {}).get("pointing_acc") or 0.0),
                "raw_output": parsed.get("raw_model_output"),
                **geometry,
                **raw_output_features(record),
            }
            candidates[interface] = candidate
            if box is not None:
                valid_boxes.append(interface)

        centers = {
            name: (candidate["pred_center_x"], candidate["pred_center_y"])
            for name, candidate in candidates.items()
            if candidate["box"] is not None
        }
        for interface, candidate in candidates.items():
            others = [name for name in INTERFACE_ORDER if name != interface and candidates[name]["box"] is not None]
            pair_ious = [box_iou(candidate["box"], candidates[name]["box"]) for name in others] if candidate["box"] is not None else [0.0]
            center_distances = []
            if interface in centers:
                cx, cy = centers[interface]
                center_distances = [math.hypot(cx - centers[name][0], cy - centers[name][1]) for name in others]
            other_areas = [candidates[name]["pred_area_ratio"] for name in others]
            candidate.update(
                {
                    "mean_pair_iou": mean(pair_ious) or 0.0,
                    "max_pair_iou": max(pair_ious) if pair_ious else 0.0,
                    "mean_center_disagreement": mean(center_distances) or 0.0,
                    "area_disagreement": mean([abs(candidate["pred_area_ratio"] - value) for value in other_areas]) or 0.0,
                    "valid_candidate_count": float(len(valid_boxes)),
                }
            )

        expression_length = float(factors.get("expression_length") or len(str(sample.get("referring_expression") or "").split()))
        samples.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "sample_id": sample_id,
                "group_id": str(sample.get("image_path") or sample_id),
                "image_path": sample.get("image_path"),
                "expression": sample.get("referring_expression"),
                "expression_length": expression_length,
                "expression_length_log": math.log1p(expression_length),
                "has_spatial_words": float(bool(factors.get("has_spatial_words"))),
                "has_relation_words": float(bool(factors.get("has_relation_words"))),
                "has_attribute_words": float(bool(factors.get("has_attribute_words"))),
                "likely_part_reference": float(bool(factors.get("likely_part_reference"))),
                "candidates": candidates,
            }
        )
    return samples, run_ids


def feature_names(full):
    names = list(EXPR_FEATURES)
    if full:
        names.extend(CANDIDATE_FEATURES)
    for interface in INTERFACE_ORDER:
        names.append(f"operator_{interface}")
        names.extend(f"{feature}_x_{interface}" for feature in EXPR_FEATURES)
    return names


def feature_vector(sample, candidate, full):
    values = [float(sample[name]) for name in EXPR_FEATURES]
    if full:
        values.extend(float(candidate[name]) for name in CANDIDATE_FEATURES)
    for interface in INTERFACE_ORDER:
        active = 1.0 if candidate["interface"] == interface else 0.0
        values.append(active)
        values.extend(active * float(sample[name]) for name in EXPR_FEATURES)
    return values


def fit_router(train_samples, full, seed):
    x, y = [], []
    for sample in train_samples:
        for interface in INTERFACE_ORDER:
            candidate = sample["candidates"][interface]
            x.append(feature_vector(sample, candidate, full))
            y.append(1 if candidate["iou"] >= 0.5 else 0)
    if len(set(y)) < 2:
        return None, float(np.mean(y))
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
    return pipeline, None


def score_candidates(model, fallback, sample, full):
    x = np.asarray(
        [feature_vector(sample, sample["candidates"][interface], full) for interface in INTERFACE_ORDER],
        dtype=float,
    )
    if model is None:
        return {interface: fallback for interface in INTERFACE_ORDER}
    probabilities = model.predict_proba(x)[:, 1]
    return {interface: float(score) for interface, score in zip(INTERFACE_ORDER, probabilities)}


def choose(scores):
    return max(INTERFACE_ORDER, key=lambda name: (scores[name], -INTERFACE_ORDER.index(name)))


def medoid_choice(sample):
    return max(
        INTERFACE_ORDER,
        key=lambda name: (
            sample["candidates"][name]["mean_pair_iou"],
            sample["candidates"][name]["parse_success"],
            -INTERFACE_ORDER.index(name),
        ),
    )


def oracle_choice(sample):
    return max(INTERFACE_ORDER, key=lambda name: (sample["candidates"][name]["iou"], -INTERFACE_ORDER.index(name)))


def grouped_predictions(samples, seed):
    groups = np.asarray([sample["group_id"] for sample in samples])
    splitter = GroupKFold(n_splits=5)
    rows = []
    fold_rows = []
    indices = np.arange(len(samples))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(indices, groups=groups)):
        train = [samples[index] for index in train_idx]
        test = [samples[index] for index in test_idx]
        fixed_means = {
            interface: mean(sample["candidates"][interface]["iou"] for sample in train)
            for interface in INTERFACE_ORDER
        }
        fold_fixed = max(INTERFACE_ORDER, key=lambda name: (fixed_means[name], -INTERFACE_ORDER.index(name)))
        factor_model, factor_fallback = fit_router(train, full=False, seed=seed + fold)
        full_model, full_fallback = fit_router(train, full=True, seed=seed + 100 + fold)
        for index in test_idx:
            sample = samples[index]
            factor_scores = score_candidates(factor_model, factor_fallback, sample, full=False)
            full_scores = score_candidates(full_model, full_fallback, sample, full=True)
            choices = {
                "fold_fixed": fold_fixed,
                "medoid": medoid_choice(sample),
                "factor_router": choose(factor_scores),
                "full_router": choose(full_scores),
                "oracle": oracle_choice(sample),
            }
            row = {
                "model_key": sample["model_key"],
                "sample_id": sample["sample_id"],
                "group_id": sample["group_id"],
                "fold": fold,
                "all_candidates_valid": int(all(sample["candidates"][name]["parse_success"] for name in INTERFACE_ORDER)),
            }
            for strategy, interface in choices.items():
                row[f"{strategy}_interface"] = interface
                row[f"{strategy}_iou"] = sample["candidates"][interface]["iou"]
                row[f"{strategy}_iou_at_0_5"] = sample["candidates"][interface]["iou_at_0_5"]
                row[f"{strategy}_pointing"] = sample["candidates"][interface]["pointing_acc"]
            for interface, score in factor_scores.items():
                row[f"factor_score_{interface}"] = score
            for interface, score in full_scores.items():
                row[f"full_score_{interface}"] = score
            rows.append(row)
            fold_rows.append(
                {
                    "model_key": sample["model_key"],
                    "sample_id": sample["sample_id"],
                    "group_id": sample["group_id"],
                    "fold": fold,
                }
            )
    return rows, fold_rows


def strategy_metrics(samples, cv_rows):
    sample_map = {sample["sample_id"]: sample for sample in samples}
    strategies = list(INTERFACE_ORDER) + ["best_fixed_global", "fold_fixed", "medoid", "factor_router", "full_router", "oracle"]
    fixed_means = {
        interface: mean(sample["candidates"][interface]["iou"] for sample in samples)
        for interface in INTERFACE_ORDER
    }
    best_fixed = max(INTERFACE_ORDER, key=lambda name: (fixed_means[name], -INTERFACE_ORDER.index(name)))
    output = []
    for population, valid_only in (("all_outputs", False), ("all_candidates_valid", True)):
        selected_rows = [row for row in cv_rows if not valid_only or row["all_candidates_valid"]]
        for strategy in strategies:
            values = []
            for row in selected_rows:
                sample = sample_map[row["sample_id"]]
                if strategy in INTERFACE_ORDER:
                    interface = strategy
                elif strategy == "best_fixed_global":
                    interface = best_fixed
                else:
                    interface = row[f"{strategy}_interface"]
                candidate = sample["candidates"][interface]
                values.append(candidate)
            output.append(
                {
                    "model_key": samples[0]["model_key"],
                    "population": population,
                    "strategy": strategy,
                    "selected_interface_if_fixed": best_fixed if strategy == "best_fixed_global" else strategy if strategy in INTERFACE_ORDER else "dynamic",
                    "n": len(values),
                    "mean_iou": mean(value["iou"] for value in values),
                    "median_iou": percentile([value["iou"] for value in values], 50),
                    "iou_at_0_5": mean(value["iou_at_0_5"] for value in values),
                    "pointing_acc": mean(value["pointing_acc"] for value in values),
                    "parse_success": mean(value["parse_success"] for value in values),
                }
            )
    return output, best_fixed


def complementarity(samples):
    unique_counts = Counter()
    tie_count = 0
    for sample in samples:
        values = {name: sample["candidates"][name]["iou"] for name in INTERFACE_ORDER}
        best = max(values.values())
        winners = [name for name, value in values.items() if abs(value - best) <= 1e-9]
        if len(winners) == 1:
            unique_counts[winners[0]] += 1
        else:
            tie_count += 1
    rows = [
        {
            "model_key": samples[0]["model_key"],
            "interface": interface,
            "unique_best_n": unique_counts[interface],
            "unique_best_fraction": unique_counts[interface] / len(samples),
        }
        for interface in INTERFACE_ORDER
    ]
    rows.append(
        {
            "model_key": samples[0]["model_key"],
            "interface": "tie",
            "unique_best_n": tie_count,
            "unique_best_fraction": tie_count / len(samples),
        }
    )
    return rows


def pairwise_rows(samples):
    rows = []
    for index, left in enumerate(INTERFACE_ORDER):
        for right in INTERFACE_ORDER[index + 1 :]:
            left_fail = [sample["candidates"][left]["iou"] < 0.5 for sample in samples]
            right_fail = [sample["candidates"][right]["iou"] < 0.5 for sample in samples]
            rows.append(
                {
                    "model_key": samples[0]["model_key"],
                    "left": left,
                    "right": right,
                    "mean_prediction_iou": mean(box_iou(sample["candidates"][left]["box"], sample["candidates"][right]["box"]) for sample in samples),
                    "both_fail": sum(a and b for a, b in zip(left_fail, right_fail)),
                    "left_only_fail": sum(a and not b for a, b in zip(left_fail, right_fail)),
                    "right_only_fail": sum(not a and b for a, b in zip(left_fail, right_fail)),
                    "both_succeed": sum(not a and not b for a, b in zip(left_fail, right_fail)),
                }
            )
    return rows


def factor_conditioned_rows(samples):
    native_areas = [sample["candidates"]["native"]["pred_area_ratio"] for sample in samples]
    area_edges = np.percentile(native_areas, [25, 50, 75])
    lengths = [sample["expression_length"] for sample in samples]
    length_edges = np.percentile(lengths, [33.333, 66.667])

    def area_bin(value):
        return "q1" if value <= area_edges[0] else "q2" if value <= area_edges[1] else "q3" if value <= area_edges[2] else "q4"

    def length_bin(value):
        return "short" if value <= length_edges[0] else "medium" if value <= length_edges[1] else "long"

    definitions = {
        "relation": lambda sample: str(bool(sample["has_relation_words"])).lower(),
        "spatial": lambda sample: str(bool(sample["has_spatial_words"])).lower(),
        "part_reference": lambda sample: str(bool(sample["likely_part_reference"])).lower(),
        "attribute": lambda sample: str(bool(sample["has_attribute_words"])).lower(),
        "expression_length": lambda sample: length_bin(sample["expression_length"]),
        "native_predicted_area": lambda sample: area_bin(sample["candidates"]["native"]["pred_area_ratio"]),
        "task_style": lambda sample: "referring",
    }
    rows = []
    for factor, grouper in definitions.items():
        grouped = defaultdict(list)
        for sample in samples:
            grouped[grouper(sample)].append(sample)
        for level, group in sorted(grouped.items()):
            winners = Counter()
            for sample in group:
                values = {name: sample["candidates"][name]["iou"] for name in INTERFACE_ORDER}
                best = max(values.values())
                best_names = [name for name, value in values.items() if abs(value - best) <= 1e-9]
                winners[best_names[0] if len(best_names) == 1 else "tie"] += 1
            rows.append(
                {
                    "model_key": samples[0]["model_key"],
                    "factor": factor,
                    "level": level,
                    "n": len(group),
                    **{f"best_{interface}_n": winners[interface] for interface in INTERFACE_ORDER},
                    **{f"best_{interface}_fraction": winners[interface] / len(group) for interface in INTERFACE_ORDER},
                    "tie_n": winners["tie"],
                    "tie_fraction": winners["tie"] / len(group),
                }
            )
    return rows


def bootstrap_strategy_deltas(samples, cv_rows, best_fixed, n_bootstrap, seed):
    sample_map = {sample["sample_id"]: sample for sample in samples}
    rng = random.Random(seed)
    rows = []
    comparisons = ("fold_fixed", "medoid", "factor_router", "full_router", "oracle")
    for population, selected in (
        ("all_outputs", cv_rows),
        ("all_candidates_valid", [row for row in cv_rows if row["all_candidates_valid"]]),
    ):
        point_deltas = {}
        for strategy in comparisons:
            point_deltas[strategy] = [
                sample_map[row["sample_id"]]["candidates"][row[f"{strategy}_interface"]]["iou"]
                - sample_map[row["sample_id"]]["candidates"][best_fixed]["iou"]
                for row in selected
            ]
        distributions = {strategy: [] for strategy in comparisons}
        for _ in range(n_bootstrap):
            sample_indices = [rng.randrange(len(selected)) for _ in range(len(selected))]
            for strategy in comparisons:
                distributions[strategy].append(mean(point_deltas[strategy][index] for index in sample_indices))
        for strategy in comparisons:
            rows.append(
                {
                    "model_key": samples[0]["model_key"],
                    "population": population,
                    "comparison": f"{strategy}_minus_best_fixed_global",
                    "n": len(selected),
                    "mean_delta_iou": mean(point_deltas[strategy]),
                    "ci95_low": percentile(distributions[strategy], 2.5),
                    "ci95_high": percentile(distributions[strategy], 97.5),
                }
            )
    return rows


def candidate_rows(samples):
    rows = []
    for sample in samples:
        for interface in INTERFACE_ORDER:
            candidate = sample["candidates"][interface]
            rows.append(
                {
                    "model_key": sample["model_key"],
                    "sample_id": sample["sample_id"],
                    "group_id": sample["group_id"],
                    "expression": sample["expression"],
                    "interface": interface,
                    "parse_success": candidate["parse_success"],
                    "iou": candidate["iou"],
                    "iou_at_0_5": candidate["iou_at_0_5"],
                    "pointing_acc": candidate["pointing_acc"],
                    **{name: sample[name] for name in EXPR_FEATURES},
                    **{name: candidate[name] for name in CANDIDATE_FEATURES},
                }
            )
    return rows


def gate_decision(summary_rows, best_fixed_by_model):
    decisions = []
    for model_key, best_fixed in best_fixed_by_model.items():
        rows = {row["strategy"]: row for row in summary_rows if row["model_key"] == model_key and row["population"] == "all_outputs"}
        valid_rows = {row["strategy"]: row for row in summary_rows if row["model_key"] == model_key and row["population"] == "all_candidates_valid"}
        fixed = rows["best_fixed_global"]
        oracle = rows["oracle"]
        router = rows["full_router"]
        oracle_gap_iou = oracle["mean_iou"] - fixed["mean_iou"]
        oracle_gap_iou50 = oracle["iou_at_0_5"] - fixed["iou_at_0_5"]
        recovery = (router["mean_iou"] - fixed["mean_iou"]) / oracle_gap_iou if oracle_gap_iou > 0 else 0.0
        valid_gap = valid_rows["oracle"]["mean_iou"] - valid_rows["best_fixed_global"]["mean_iou"]
        valid_recovery = (
            (valid_rows["full_router"]["mean_iou"] - valid_rows["best_fixed_global"]["mean_iou"]) / valid_gap
            if valid_gap > 0
            else 0.0
        )
        headroom_pass = oracle_gap_iou >= 0.025 or oracle_gap_iou50 >= 0.035
        router_pass = recovery >= 0.25
        parse_independent = valid_gap >= 0.025 or (
            valid_rows["oracle"]["iou_at_0_5"] - valid_rows["best_fixed_global"]["iou_at_0_5"] >= 0.035
        )
        decisions.append(
            {
                "model_key": model_key,
                "best_fixed_interface": best_fixed,
                "oracle_gap_mean_iou": oracle_gap_iou,
                "oracle_gap_iou_at_0_5": oracle_gap_iou50,
                "full_router_delta_mean_iou": router["mean_iou"] - fixed["mean_iou"],
                "full_router_recovery_fraction": recovery,
                "valid_only_oracle_gap_mean_iou": valid_gap,
                "valid_only_router_recovery_fraction": valid_recovery,
                "headroom_pass": headroom_pass,
                "router_pass": router_pass,
                "parse_independent_headroom": parse_independent,
                "model_gate_pass": bool(headroom_pass and router_pass and parse_independent),
            }
        )
    return decisions


def render_report(path, summary_rows, decisions, complementarity_rows, factor_rows, pair_rows, run_ids):
    lines = [
        "# FAR-Ground M1 Headroom Report\n\n",
        "This report evaluates existing matched RefCOCO val 1000 predictions only. No new model inference was run. "
        "The router uses image-grouped out-of-fold predictions and test-time-legal features.\n\n",
        "## Strategy Results\n\n",
        "| Model | Population | Strategy | n | Parse | mIoU | IoU@0.5 | Pointing |\n",
        "|---|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_key']} | {row['population']} | {row['strategy']} | {row['n']} | "
            f"{row['parse_success']:.3f} | {row['mean_iou']:.4f} | {row['iou_at_0_5']:.4f} | {row['pointing_acc']:.4f} |\n"
        )
    lines.extend(
        [
            "\n## Hard Gate\n\n",
            "| Model | Best fixed | Oracle gap mIoU | Oracle gap IoU@0.5 | Router delta mIoU | Gap recovered | Valid-only oracle gap | Gate |\n",
            "|---|---|---:|---:|---:|---:|---:|---|\n",
        ]
    )
    for row in decisions:
        lines.append(
            f"| {row['model_key']} | {row['best_fixed_interface']} | {row['oracle_gap_mean_iou']:.4f} | "
            f"{row['oracle_gap_iou_at_0_5']:.4f} | {row['full_router_delta_mean_iou']:.4f} | "
            f"{row['full_router_recovery_fraction']:.3f} | {row['valid_only_oracle_gap_mean_iou']:.4f} | "
            f"{'PASS' if row['model_gate_pass'] else 'FAIL'} |\n"
        )
    overall_pass = any(row["model_gate_pass"] for row in decisions)
    lines.append(f"\nOverall M1 decision: **{'PASS' if overall_pass else 'FAIL'}**.\n\n")
    if overall_pass:
        lines.append("At least one model satisfies the preregistered headroom, grouped-router recovery, and parse-independent headroom conditions. M2 may proceed only for operators frozen before inference.\n")
    else:
        lines.append("The interface-routing gate is not met. Do not build or promote a large FAR-Ground router. The only allowed next method test is the preregistered maximum-200-sample factor-specific repair pilot.\n")
    lines.extend(["\n## Candidate Complementarity\n\n", "| Model | Interface | Unique best n | Fraction |\n", "|---|---|---:|---:|\n"])
    for row in complementarity_rows:
        lines.append(f"| {row['model_key']} | {row['interface']} | {row['unique_best_n']} | {row['unique_best_fraction']:.3f} |\n")
    lines.extend(["\n## Pairwise Error and Agreement\n\n", "| Model | Pair | Prediction IoU | Both fail | Left-only fail | Right-only fail | Both succeed |\n", "|---|---|---:|---:|---:|---:|---:|\n"])
    for row in pair_rows:
        lines.append(
            f"| {row['model_key']} | {row['left']} / {row['right']} | {row['mean_prediction_iou']:.3f} | "
            f"{row['both_fail']} | {row['left_only_fail']} | {row['right_only_fail']} | {row['both_succeed']} |\n"
        )
    lines.extend(
        [
            "\n## Factor Predictability\n\n",
            "`factor_conditioned_best.csv` records best-interface frequencies by relation, spatial, part, attribute, expression-length, and native-predicted-area proxies. These are associations, not causal operator effects.\n\n",
            "## Provenance\n\n",
            f"Run IDs: `{json.dumps(run_ids, sort_keys=True)}`.\n\n",
            "The router never receives GT object size, GT location, GT box, or IoU-derived features at test time. Oracle and candidate IoU columns are analysis labels only.\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples = {}
    all_run_ids = {}
    all_cv_rows = []
    all_fold_rows = []
    summary_rows = []
    complementarity_rows = []
    pair_rows = []
    factor_rows = []
    bootstrap_rows = []
    best_fixed_by_model = {}

    for spec in args.model:
        model_key, model_name, native_run, native_path, json_run, json_path, tuple_path = spec
        samples, run_ids = load_matched_model(
            model_key,
            model_name,
            (
                ("native", native_run, native_path),
                ("norm_json", json_run, json_path),
                ("tuple", "", tuple_path),
            ),
        )
        cv_rows, fold_rows = grouped_predictions(samples, args.seed)
        model_summary, best_fixed = strategy_metrics(samples, cv_rows)
        all_samples[model_key] = samples
        all_run_ids[model_key] = run_ids
        all_cv_rows.extend(cv_rows)
        all_fold_rows.extend(fold_rows)
        summary_rows.extend(model_summary)
        best_fixed_by_model[model_key] = best_fixed
        complementarity_rows.extend(complementarity(samples))
        pair_rows.extend(pairwise_rows(samples))
        factor_rows.extend(factor_conditioned_rows(samples))
        bootstrap_rows.extend(bootstrap_strategy_deltas(samples, cv_rows, best_fixed, args.bootstrap, args.seed))

    decisions = gate_decision(summary_rows, best_fixed_by_model)
    write_csv(output_dir / "fold_assignments.csv", all_fold_rows)
    write_csv(output_dir / "strategy_predictions.csv", all_cv_rows)
    write_csv(output_dir / "strategy_summary.csv", summary_rows)
    write_csv(output_dir / "candidate_complementarity.csv", complementarity_rows)
    write_csv(output_dir / "pairwise_error_overlap.csv", pair_rows)
    write_csv(output_dir / "factor_conditioned_best.csv", factor_rows)
    write_csv(output_dir / "bootstrap.csv", bootstrap_rows)
    write_csv(output_dir / "gate_decision.csv", decisions)
    write_csv(output_dir / "candidates.csv", [row for samples in all_samples.values() for row in candidate_rows(samples)])

    metrics = {
        "stage": "M1",
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "models": list(all_samples),
        "n_per_model": {key: len(value) for key, value in all_samples.items()},
        "run_ids": all_run_ids,
        "gate": decisions,
        "overall_gate_pass": any(row["model_gate_pass"] for row in decisions),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "command": "scripts/analyze_far_ground_headroom.py",
        "git_branch": "far-ground-method",
        "frozen_diagnosis_commit": "892812e",
        "protocol": "METHOD_EXPERIMENT_PLAN.md#2026-07-15-far-ground-preregistration",
        "feature_names_factor": feature_names(False),
        "feature_names_full": feature_names(True),
        "grouping": "image_path",
        "n_splits": 5,
        "estimator": "StandardScaler + LogisticRegression(C=1.0,class_weight=balanced,max_iter=5000)",
        "run_ids": all_run_ids,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "config.yaml").write_text(
        "stage: M1\nseed: 20260715\nbootstrap: 5000\npopulation: refcoco_val_matched_1000\ngroup_by: image_path\ninterfaces:\n  - native\n  - norm_json\n  - tuple\n",
        encoding="utf-8",
    )
    report_path = output_dir / "METHOD_HEADROOM_REPORT.md"
    render_report(report_path, summary_rows, decisions, complementarity_rows, factor_rows, pair_rows, all_run_ids)
    (REPO_ROOT / "METHOD_HEADROOM_REPORT.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "gate": decisions}, indent=2))


if __name__ == "__main__":
    main()
