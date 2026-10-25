"""CPU tests for the SafeGround-derived score (run: python r9_1/tests/test_r9_score.py).

Expected values are derived by hand from the vendored formulas:
  U = 0.6*CD + 0.2*IE + 0.2*TA ; M = number of 4-connected regions of active patches.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r9_score as S  # noqa: E402

FAILS = []


def close(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got:.6f} want {want:.6f}")
    if not ok:
        FAILS.append(name)


def box_at(x, y, half=2.0):
    return [x - half, y - half, x + half, y + half]


W = H = 1000          # original image
RW = RH = 560         # resized frame -> 20 x 20 patches of 28 px (code default)


def run(boxes):
    return S.score_input(boxes, W, H, RW, RH)


# 1) all ten samples in one patch: M=1 -> TA=max(0.1,1-1)=0.1, IE=0.5, CD=0.1
r = run([box_at(100, 100)] * 10)
close("one patch: combined", r["code_default_combined"], 0.6 * 0.1 + 0.2 * 0.5 + 0.2 * 0.1)
close("one patch: margin(TA)", r["code_default_margin"], 0.1)

# 2) two separated patches, 5/5 split: S1=S2=0.5 -> TA=1, IE=1, CD=0.5
r = run([box_at(100, 100)] * 5 + [box_at(800, 800)] * 5)
close("two equal regions: combined", r["code_default_combined"], 0.6 * 0.5 + 0.2 * 1 + 0.2 * 1, 1e-6)

# 3) two separated patches, 8/2 split: S=(0.8,0.2) -> TA=1-0.6/0.8=0.25,
#    p=(0.8,0.2) -> IE=-(0.8ln0.8+0.2ln0.2)/ln2, CD=1-(0.64+0.04)=0.32
ie = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2)) / math.log(2)
r = run([box_at(100, 100)] * 8 + [box_at(800, 800)] * 2)
close("8/2 regions: combined", r["code_default_combined"], 0.6 * 0.32 + 0.2 * ie + 0.2 * 0.25, 1e-6)

# 4) two ADJACENT patches (4-connected) form one region: mean P=0.5 -> M=1, TA=max(0.1,0.5)=0.5
r = run([box_at(100, 100)] * 5 + [box_at(100 + 28 * 1000 / 560, 100)] * 5)
close("adjacent patches merge: combined", r["code_default_combined"], 0.6 * 0.1 + 0.2 * 0.5 + 0.2 * 0.5)

# 5) diagonal neighbours are NOT 4-connected -> two equal regions
d = 28 * 1000 / 560
r = run([box_at(100, 100)] * 5 + [box_at(100 + d, 100 + d)] * 5)
close("diagonal patches stay separate: combined", r["code_default_combined"], 0.6 * 0.5 + 0.2 * 1 + 0.2 * 1)

# 6) ten mutually separated patches, one sample each: TA=1, IE=1, CD=0.9
r = run([box_at(30 + 90 * i, 30 + 90 * i) for i in range(10)])
close("ten separate regions: combined", r["code_default_combined"], 0.6 * 0.9 + 0.2 + 0.2, 1e-6)

# 7) all samples invalid -> empty heatmap -> every component 1.0
r = run([None] * 10)
close("all invalid: combined", r["code_default_combined"], 1.0)
assert r["n_valid"] == 0

# 8) partial validity: invalid samples are skipped (distribution over valid ones)
r = run([box_at(100, 100)] * 3 + [None] * 7)
close("3 valid in one patch: combined", r["code_default_combined"], 0.6 * 0.1 + 0.2 * 0.5 + 0.2 * 0.1)
assert r["n_valid"] == 3

# 9) mapping uses the resized frame: same normalized centers, different resized sizes
a = S.score_input([box_at(100, 100)] * 5 + [box_at(130, 100)] * 5, W, H, 560, 560)   # x=56 and 72.8 px -> same patch (col 2)
b = S.score_input([box_at(100, 100)] * 5 + [box_at(130, 100)] * 5, W, H, 2800, 2800)  # x=280 and 364 px -> cols 10 and 13
close("resize frame: same patch at 560", a["code_default_combined"], 0.6 * 0.1 + 0.2 * 0.5 + 0.2 * 0.1)
close("resize frame changes connectivity (2800)", b["code_default_combined"], 0.6 * 0.5 + 0.2 * 1 + 0.2 * 1)

# 10) paper-parameter variant: threshold 0.3 drops a weak region (8/2 -> 2/10=0.2 < 0.3*0.8)
r = run([box_at(100, 100)] * 8 + [box_at(800, 800)] * 2)
close("paper variant drops weak region", r["paper_params_combined"], 0.6 * 0.1 + 0.2 * 0.5 + 0.2 * 0.2)

# 11) score is bounded and order-invariant
import random  # noqa: E402
rng = random.Random(0)
for t in range(200):
    bx = [box_at(rng.uniform(0, 1000), rng.uniform(0, 1000)) if rng.random() > 0.2 else None for _ in range(10)]
    r1, r2 = run(bx), run(list(reversed(bx)))
    assert 0.0 <= r1["code_default_combined"] <= 1.0
    assert abs(r1["code_default_combined"] - r2["code_default_combined"]) < 1e-12
print("PASS  bounds and order invariance over 200 random inputs")

print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL'}")
sys.exit(1 if FAILS else 0)
