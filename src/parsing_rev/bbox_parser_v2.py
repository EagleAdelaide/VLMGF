"""Corrected bbox parser (WWW 2027 revision, P0-4).

Changes vs src/parsing/bbox_parser.py (defects confirmed 2026-10-09 on HEAD 8460dde):
  D1  Explicit coordinate_space now takes precedence over the <=1.5 fractional
      heuristic. Under "pixel", [0,0,1,1] is a 1x1-pixel box (previously: scaled
      to the full image). Under "norm1000", values <=1.5 are divided by 1000
      (previously: multiplied by image size). "auto" keeps the old heuristic.
  D2  Non-finite coordinates (NaN/Inf) are rejected before scaling/clipping
      (previously: [-Inf,0,Inf,400] survived clipping as a valid full-width box).
  D3  Point->box expansion is flagged with parse_point_expanded=True so that
      box-interface runs that silently fell through to the point branch can be
      counted and excluded/handled explicitly.

Everything else (candidate extraction order, recovery rule, confidence
extraction) is byte-equivalent in behaviour to the original module, which this
module imports for the extraction helpers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.parsing import bbox_parser as _orig  # noqa: E402

_recover_dropped_decimal_bbox = _orig._recover_dropped_decimal_bbox if hasattr(_orig, "_recover_dropped_decimal_bbox") else None


def _all_finite(values):
    return all(math.isfinite(float(v)) for v in values)


def normalize_or_scale_bbox(bbox, image_width, image_height, coordinate_space="auto"):
    values = [float(v) for v in bbox]
    if not _all_finite(values):  # D2
        return None
    if coordinate_space == "norm1000":  # D1: explicit space first
        max_value = max(abs(v) for v in values)
        if max_value > 1000.0 * 1.05:
            return None
        return [
            values[0] / 1000.0 * image_width,
            values[1] / 1000.0 * image_height,
            values[2] / 1000.0 * image_width,
            values[3] / 1000.0 * image_height,
        ]
    if coordinate_space == "pixel":  # D1
        return values
    # auto: original heuristic unchanged
    max_value = max(abs(v) for v in values)
    if max_value <= 1.5:
        return [values[0] * image_width, values[1] * image_height, values[2] * image_width, values[3] * image_height]
    pixel_tolerance = 1.05
    clearly_exceeds_pixels = values[2] > float(image_width) * pixel_tolerance or values[3] > float(image_height) * pixel_tolerance
    if max_value <= 1000.0 and clearly_exceeds_pixels:
        return [values[0] / 1000.0 * image_width, values[1] / 1000.0 * image_height, values[2] / 1000.0 * image_width, values[3] / 1000.0 * image_height]
    return values


def normalize_or_scale_point(point, image_width, image_height, coordinate_space="auto"):
    x, y = [float(v) for v in point]
    if not _all_finite([x, y]):  # D2
        return None
    if coordinate_space == "norm100":
        if max(abs(x), abs(y)) > 100.0 * 1.05:
            return None
        return [x / 100.0 * image_width, y / 100.0 * image_height]
    if coordinate_space == "norm1000":  # D1
        if max(abs(x), abs(y)) > 1000.0 * 1.05:
            return None
        return [x / 1000.0 * image_width, y / 1000.0 * image_height]
    if coordinate_space == "pixel":  # D1
        return [x, y]
    max_value = max(abs(x), abs(y))
    if max_value <= 1.5:
        return [x * image_width, y * image_height]
    pixel_tolerance = 1.05
    clearly_exceeds_pixels = x > float(image_width) * pixel_tolerance or y > float(image_height) * pixel_tolerance
    if max_value <= 1000.0 and clearly_exceeds_pixels:
        return [x / 1000.0 * image_width, y / 1000.0 * image_height]
    return [x, y]


def clip_and_order_bbox(bbox, image_width, image_height):
    values = [float(v) for v in bbox]
    if not _all_finite(values):  # D2 (defence in depth)
        return None
    return _orig.clip_and_order_bbox(values, image_width, image_height)


def _finish(result, bbox, confidence, image_width, image_height, source, coordinate_space="auto", recover_dropped_decimal=False):
    converted = normalize_or_scale_bbox(bbox, image_width, image_height, coordinate_space)
    recovered_flag = False
    if converted is None and recover_dropped_decimal and coordinate_space == "norm1000":
        # Original recovery rule verbatim, plus the D2 finite guard.
        values = [float(v) for v in bbox]
        if _all_finite(values) and max(abs(v) for v in values) <= 10500.0:
            recovered = [v / 10.0 if abs(v) > 1050.0 else v for v in values]
            converted = normalize_or_scale_bbox(recovered, image_width, image_height, coordinate_space)
            recovered_flag = converted is not None
    if converted is None:
        result["parse_error_type"] = "coordinates_out_of_range"
        return result
    clipped = clip_and_order_bbox(converted, image_width, image_height)
    if clipped is None:
        result["parse_error_type"] = "invalid_box_after_clipping"
        return result
    result["parsed_bbox_xyxy"] = clipped
    result["parse_success"] = True
    result["parse_error_type"] = None
    result["confidence"] = confidence
    result["parse_source"] = source
    result["parse_recovery"] = "dropped_decimal" if recovered_flag else None
    return result


def _finish_point(result, point, confidence, image_width, image_height, source, coordinate_space="auto"):
    converted_point = normalize_or_scale_point(point, image_width, image_height, coordinate_space)
    if converted_point is None:
        result["parse_error_type"] = "coordinates_out_of_range"
        return result
    x, y = converted_point
    side = max(8.0, min(float(image_width), float(image_height)) * 0.05)
    bbox = [x - side / 2.0, y - side / 2.0, x + side / 2.0, y + side / 2.0]
    clipped = clip_and_order_bbox(bbox, image_width, image_height)
    if clipped is None:
        result["parse_error_type"] = "invalid_point_after_clipping"
        return result
    result["parsed_bbox_xyxy"] = clipped
    result["parsed_point_xy"] = [
        max(0.0, min(float(image_width), float(x))),
        max(0.0, min(float(image_height), float(y))),
    ]
    result["parse_success"] = True
    result["parse_error_type"] = None
    result["confidence"] = confidence
    result["parse_source"] = source
    result["parse_point_expanded"] = True  # D3
    return result


def parse_bbox_output(raw_text, image_width, image_height, coordinate_space="auto", recover_dropped_decimal=False):
    """Same cascade as the original parser, with corrected finishing functions."""
    result = {
        "parse_success": False,
        "parsed_bbox_xyxy": None,
        "parsed_point_xy": None,
        "confidence": None,
        "parse_error_type": "no_bbox_found",
        "parse_source": None,
        "parse_recovery": None,
        "parse_point_expanded": False,
        "raw_model_output": raw_text,
    }
    if not raw_text or not str(raw_text).strip():
        result["parse_error_type"] = "empty_output"
        return result
    text = str(raw_text)

    for candidate in _orig._json_candidates(text):
        obj = _orig._try_json_candidate(candidate)
        if obj is None:
            continue
        bbox = _orig._extract_bbox_from_object(obj)
        confidence = _orig._extract_confidence(obj)
        if bbox is not None:
            return _finish(result, bbox, confidence, image_width, image_height, "json", coordinate_space, recover_dropped_decimal)
        point = _orig._extract_point_from_object(obj)
        if point is not None:
            return _finish_point(result, point, confidence, image_width, image_height, "json_point", coordinate_space)

    molmo_point = _orig._extract_molmo_point(text)
    if molmo_point is not None:
        return _finish_point(result, molmo_point, None, image_width, image_height, "molmo_point", coordinate_space)

    box_tag_bbox = _orig._extract_box_tag(text)
    if box_tag_bbox is not None:
        confidence = _orig._extract_confidence_from_text(text)
        return _finish(result, box_tag_bbox, confidence, image_width, image_height, "ref_box_tag", coordinate_space, recover_dropped_decimal)

    list_bbox = _orig._extract_coordinate_list(text)
    if list_bbox is not None:
        confidence = _orig._extract_confidence_from_text(text)
        return _finish(result, list_bbox, confidence, image_width, image_height, "coordinate_list", coordinate_space, recover_dropped_decimal)

    point = _orig._extract_point(text)
    if point is not None:
        confidence = _orig._extract_confidence_from_text(text)
        return _finish_point(result, point, confidence, image_width, image_height, "point_recovered", coordinate_space)

    return result
