"""Robust bbox parser for VLM grounding outputs."""

import json
import re


def parse_bbox_output(raw_text, image_width, image_height, coordinate_space="auto", recover_dropped_decimal=False):
    """Parse a raw model string into a pixel-space bbox.

    coordinate_space declares the model's coordinate convention:
    - "auto": heuristic (rescale 0-1000 values only when they clearly exceed pixels)
    - "norm1000": always interpret coordinates as 0-1000 normalized (InternVL-style);
      values beyond 1000*1.05 are rejected as out-of-range instead of clipped
    - "pixel": always interpret as pixels

    recover_dropped_decimal (norm1000 only): out-of-range values in (1050, 10500] are
    retried divided by 10 (dropped-decimal-point decoding defect). Recovered records
    are marked with parse_recovery="dropped_decimal" so analyses can separate them.
    """
    result = {
        "parsed_bbox_xyxy": None,
        "parsed_point_xy": None,
        "parse_success": False,
        "parse_error_type": None,
        "parse_recovery": None,
        "confidence": None,
        "raw_model_output": raw_text,
    }
    if raw_text is None:
        result["parse_error_type"] = "empty_output"
        return result
    text = str(raw_text).strip()
    if not text:
        result["parse_error_type"] = "empty_output"
        return result

    json_candidates = _json_candidates(text)
    for candidate in json_candidates:
        parsed = _try_json_candidate(candidate)
        if parsed is None:
            continue
        bbox = _extract_bbox_from_object(parsed)
        confidence = _extract_confidence(parsed)
        if bbox is not None:
            return _finish(result, bbox, confidence, image_width, image_height, "json", coordinate_space, recover_dropped_decimal)
        point = _extract_point_from_object(parsed)
        if point is not None:
            return _finish_point(result, point, confidence, image_width, image_height, "json_point", coordinate_space)

    molmo_point = _extract_molmo_point(text)
    if molmo_point is not None:
        return _finish_point(result, molmo_point, None, image_width, image_height, "molmo_point", coordinate_space)

    box_tag_bbox = _extract_box_tag(text)
    if box_tag_bbox is not None:
        confidence = _extract_confidence_from_text(text)
        return _finish(result, box_tag_bbox, confidence, image_width, image_height, "ref_box_tag", coordinate_space, recover_dropped_decimal)

    list_bbox = _extract_coordinate_list(text)
    if list_bbox is not None:
        confidence = _extract_confidence_from_text(text)
        return _finish(result, list_bbox, confidence, image_width, image_height, "coordinate_list", coordinate_space, recover_dropped_decimal)

    point = _extract_point(text)
    if point is not None:
        confidence = _extract_confidence_from_text(text)
        return _finish_point(result, point, confidence, image_width, image_height, "point_recovered", coordinate_space)

    result["parse_error_type"] = "no_recoverable_coordinates"
    return result


def _json_candidates(text):
    candidates = []
    list_fence_matches = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(list_fence_matches)
    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fence_matches)
    object_matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    candidates.extend(object_matches)
    if text.startswith("[") and text.endswith("]"):
        candidates.append(text)
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    return candidates


def _try_json_candidate(candidate):
    try:
        return json.loads(candidate)
    except Exception:
        repaired = candidate.replace("'", '"')
        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)
        try:
            return json.loads(repaired)
        except Exception:
            return None


def _extract_bbox_from_object(obj):
    if isinstance(obj, list):
        for item in obj:
            bbox = _extract_bbox_from_object(item)
            if bbox is not None:
                return bbox
        return None
    if not isinstance(obj, dict):
        return None
    for key in ["bbox", "box", "bounding_box", "bbox_xyxy", "bbox_2d", "box_2d"]:
        value = obj.get(key)
        if _is_four_numbers(value):
            return [float(v) for v in value]
    if all(k in obj for k in ["x1", "y1", "x2", "y2"]):
        return [float(obj[k]) for k in ["x1", "y1", "x2", "y2"]]
    return None


def _extract_point_from_object(obj):
    if isinstance(obj, list):
        for item in obj:
            point = _extract_point_from_object(item)
            if point is not None:
                return point
        return None
    if not isinstance(obj, dict):
        return None
    for key in ["point", "center", "point_xy", "point_2d", "center_2d"]:
        value = obj.get(key)
        if _is_two_numbers(value):
            return [float(v) for v in value]
    if all(k in obj for k in ["x", "y"]):
        return [float(obj["x"]), float(obj["y"])]
    return None


def _extract_confidence(obj):
    if isinstance(obj, list):
        for item in obj:
            confidence = _extract_confidence(item)
            if confidence is not None:
                return confidence
        return None
    if not isinstance(obj, dict):
        return None
    for key in ["confidence", "score", "probability"]:
        if key in obj:
            try:
                return max(0.0, min(1.0, float(obj[key])))
            except Exception:
                return None
    return None


def _is_four_numbers(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        [float(v) for v in value]
        return True
    except Exception:
        return False


def _is_two_numbers(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        [float(v) for v in value]
        return True
    except Exception:
        return False


def _extract_coordinate_list(text):
    labelled = re.search(
        r"(?:bbox|box|coordinates?)\D+(-?\d+(?:\.\d+)?)\D+(-?\d+(?:\.\d+)?)\D+(-?\d+(?:\.\d+)?)\D+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if labelled:
        return [float(v) for v in labelled.groups()]
    bracket = re.search(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        text,
    )
    if bracket:
        return [float(v) for v in bracket.groups()]
    tuple_match = re.search(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        text,
    )
    if tuple_match:
        return [float(v) for v in tuple_match.groups()]
    return None


def _extract_point(text):
    match = re.search(
        r"(?:point|center)(?!_\d+d)\D+(-?\d+(?:\.\d+)?)\D+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return [float(v) for v in match.groups()]
    return None


def _extract_molmo_point(text):
    """Extract the first Molmo point from old and Molmo2 point tags.

    Molmo-7B-D emits percentage coordinates as <point x=".." y=".."> or
    <points x1=".." y1="..">. Molmo2 pointing examples emit 0-1000 coordinates inside a
    coords attribute, e.g. <points coords="0 1 123 456"/>. The caller's
    coordinate_space controls how the returned point is scaled.
    """
    if "<point" not in text:
        return None
    single = re.search(r'<point\b[^>]*\bx\s*=\s*"([-\d.]+)"[^>]*\by\s*=\s*"([-\d.]+)"', text)
    if single:
        return [float(single.group(1)), float(single.group(2))]
    multi = re.search(r'<points\b[^>]*\bx1\s*=\s*"([-\d.]+)"[^>]*\by1\s*=\s*"([-\d.]+)"', text)
    if multi:
        return [float(multi.group(1)), float(multi.group(2))]
    coords = re.search(r'<points\b[^>]*\bcoords\s*=\s*"([0-9\t:;, .-]+)"', text)
    if coords:
        # Molmo2 examples encode points as "frame id x y" groups where x/y are
        # zero-padded 0-1000 coordinates, often nested as "frame id x y". For image
        # prompts, accepting the first encoded point gives a point-native output without
        # manufacturing a box.
        groups = re.split(r"[\t:;,]+", coords.group(1))
        for group in groups:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", group)
            if len(numbers) >= 4:
                return [float(numbers[2]), float(numbers[3])]
            if len(numbers) == 3:
                return [float(numbers[1]), float(numbers[2])]
            if len(numbers) == 2:
                return [float(numbers[0]), float(numbers[1])]
    return None


def _extract_box_tag(text):
    """Extract InternVL-style <box>[[x1, y1, x2, y2]]</box> coordinates.

    The captured tokens may be malformed (e.g. a decoding glitch like "7-773" with an
    interior hyphen); such tokens are treated as an unparseable box (return None) rather
    than raising, so a single bad sample does not crash an entire evaluation run.
    """
    match = re.search(r"<box>\s*\[?\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\]?\s*</box>", text)
    if not match:
        return None
    try:
        return [float(match.group(i)) for i in range(1, 5)]
    except (ValueError, TypeError):
        return None


def _extract_confidence_from_text(text):
    match = re.search(r"(?:confidence|score|probability)\D+([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group(1))))


def _finish(result, bbox, confidence, image_width, image_height, source, coordinate_space="auto", recover_dropped_decimal=False):
    converted = normalize_or_scale_bbox(bbox, image_width, image_height, coordinate_space)
    if converted is None and recover_dropped_decimal and coordinate_space == "norm1000":
        values = [float(v) for v in bbox]
        if max(abs(v) for v in values) <= 10500.0:
            recovered = [v / 10.0 if abs(v) > 1050.0 else v for v in values]
            converted = normalize_or_scale_bbox(recovered, image_width, image_height, coordinate_space)
            if converted is not None:
                result["parse_recovery"] = "dropped_decimal"
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
    return result


def normalize_or_scale_bbox(bbox, image_width, image_height, coordinate_space="auto"):
    values = [float(v) for v in bbox]
    max_value = max(abs(v) for v in values)
    if max_value <= 1.5:
        return [values[0] * image_width, values[1] * image_height, values[2] * image_width, values[3] * image_height]
    if coordinate_space == "norm1000":
        if max_value > 1000.0 * 1.05:
            return None
        return [values[0] / 1000.0 * image_width, values[1] / 1000.0 * image_height, values[2] / 1000.0 * image_width, values[3] / 1000.0 * image_height]
    if coordinate_space == "pixel":
        return values
    pixel_tolerance = 1.05
    clearly_exceeds_pixels = values[2] > float(image_width) * pixel_tolerance or values[3] > float(image_height) * pixel_tolerance
    if max_value <= 1000.0 and clearly_exceeds_pixels:
        return [values[0] / 1000.0 * image_width, values[1] / 1000.0 * image_height, values[2] / 1000.0 * image_width, values[3] / 1000.0 * image_height]
    return values


def normalize_or_scale_point(point, image_width, image_height, coordinate_space="auto"):
    x, y = [float(v) for v in point]
    max_value = max(abs(x), abs(y))
    if coordinate_space == "norm100":
        if max_value > 100.0 * 1.05:
            return None
        return [x / 100.0 * image_width, y / 100.0 * image_height]
    if max_value <= 1.5:
        return [x * image_width, y * image_height]
    if coordinate_space == "norm1000":
        if max_value > 1000.0 * 1.05:
            return None
        return [x / 1000.0 * image_width, y / 1000.0 * image_height]
    if coordinate_space == "pixel":
        return [x, y]
    pixel_tolerance = 1.05
    clearly_exceeds_pixels = x > float(image_width) * pixel_tolerance or y > float(image_height) * pixel_tolerance
    if max_value <= 1000.0 and clearly_exceeds_pixels:
        return [x / 1000.0 * image_width, y / 1000.0 * image_height]
    return [x, y]


def clip_and_order_bbox(bbox, image_width, image_height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    left = max(0.0, min(float(image_width), left))
    right = max(0.0, min(float(image_width), right))
    top = max(0.0, min(float(image_height), top))
    bottom = max(0.0, min(float(image_height), bottom))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]
