"""Fixed tie-case regression test of the frozen threshold rule (no data, no GPU).

The R9.1 protocol (item 8) and r9_evaluate.frozen_threshold accept an input iff its failure
score is <= the calibration median, so inputs tied AT the median are accepted and the
calibration coverage can exceed 50%. This test pins that behaviour; it must not be changed to
match a different caption. Run: python r9_1/tests/test_r9_threshold_ties.py
(works in the repository, with scripts_rev/r9_evaluate.py, and in the artifact, with
r9_1/r9_evaluate.py).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in [HERE.parent, HERE.parents[1] / "scripts_rev"]:  # artifact layout, repository layout
    if (p / "r9_evaluate.py").exists():
        sys.path.insert(0, str(p))
        break
import r9_evaluate as E  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


# Calibration: 10 scores, 6 tied at 0.26 (the median); test: 5 scores, 2 tied at 0.26.
cal = np.array([0.10, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.40, 0.70, 0.90])
cal_y = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1])
test = np.array([0.05, 0.26, 0.26, 0.50, 0.95])
test_y = np.array([0, 0, 1, 1, 1])
r = E.frozen_threshold(cal, test, cal_y, test_y)
check(f"tau is the calibration median ({r['tau']})", r["tau"] == 0.26)
check(f"ties at the median are accepted: calibration coverage 0.7 ({r['calib_coverage']})", r["calib_coverage"] == 0.7)
check(f"calibration coverage exceeds 50% under ties", r["calib_coverage"] > 0.5)
check(f"calibration accepted risk 1/7 ({r['calib_accepted_risk']:.4f})", abs(r["calib_accepted_risk"] - 1 / 7) < 1e-12)
check(f"test coverage 3/5 ({r['test_coverage']})", r["test_coverage"] == 0.6)
check(f"test accepted risk 1/3 ({r['test_accepted_risk']:.4f})", abs(r["test_accepted_risk"] - 1 / 3) < 1e-12)
strict = float((cal < r["tau"]).mean())
check(f"a strict '<' rule would give a different coverage (0.1 vs 0.7): rule is '<=' ({strict})", strict == 0.1)
print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL'}")
sys.exit(1 if FAILS else 0)
