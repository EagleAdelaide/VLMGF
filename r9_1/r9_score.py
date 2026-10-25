"""SafeGround-derived score-only adaptation for frozen box predictions (R9.1).

The score is computed by the vendored, unmodified SafeGround code
(third_party/SAFEGROUND, commit 5e8fca7) from the centers of the K sampled boxes,
expressed in the coordinate frame of the image the model actually saw (the resized
image given by the processor's image_grid_thw). The frozen greedy box is not changed;
the score only ranks it. No ground truth is used.

Primary parameters = official code defaults (patch 28, activation threshold 0.0);
secondary, pre-specified variant = paper values (patch 14, activation threshold 0.3).
"""
import sys
from pathlib import Path

_TP = Path(__file__).resolve().parent / "third_party"
for p in (_TP, _TP / "SAFEGROUND" / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from uncertainty import compute_all_uncertainties  # noqa: E402  (vendored SafeGround)

VARIANTS = {
    "code_default": dict(patch_size=28, activation_threshold=0.0),
    "paper_params": dict(patch_size=14, activation_threshold=0.3),
}


def centers_in_resized_frame(boxes_xyxy, image_width, image_height, resized_width, resized_height):
    """Box centers (original-image pixels) -> (x, y) in the resized image; None for invalid boxes."""
    pts = []
    for b in boxes_xyxy:
        if b is None:
            pts.append((None, None))
            continue
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        pts.append((cx / image_width * resized_width, cy / image_height * resized_height))
    return pts


def score_input(boxes_xyxy, image_width, image_height, resized_width, resized_height):
    """Return dict with the combined score and components for both pre-specified variants.

    Invalid samples (None) are skipped by the vendored heatmap code, so the distribution
    is over valid samples only; with zero valid samples every component is 1.0.
    """
    pts = centers_in_resized_frame(boxes_xyxy, image_width, image_height, resized_width, resized_height)
    out = {"n_samples": len(boxes_xyxy), "n_valid": sum(p[0] is not None for p in pts)}
    for name, kw in VARIANTS.items():
        u = compute_all_uncertainties(pts, int(resized_width), int(resized_height), **kw)
        for k, v in u.items():
            out[f"{name}_{k}"] = float(v)
    return out
