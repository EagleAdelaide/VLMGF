"""CPU smoke: verify inventory and recompute principal Web claims from frozen rows."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts_rev'))
from entropy_scale import load_scores, failure_score, correction_root


def main():
    assert sys.version_info[:2] == (3, 11)
    for line in (ROOT / 'requirements.lock.txt').read_text().splitlines():
        if '==' in line:
            name, version = line.split('==')
            assert importlib.metadata.version(name) == version, name
    count = 0
    for line in (ROOT / 'checksums.sha256').read_text().splitlines():
        value, name = line.split(maxsplit=1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, name
        count += 1
    c = correction_root(ROOT)
    counts = pd.read_csv(c / 'p0/p0b_outcome_contingency_852.csv').set_index('model')
    accepted = pd.read_csv(c / 'p0/p0b_accepted_ids_geom_token_frozen.csv')
    risks = pd.read_csv(c / 'p0/p0b_same_accepted_set_two_outcomes.csv').set_index('model')
    points = pd.read_csv(c / 'corrected_raw_entropy_operating_points.csv')
    singles = pd.read_csv(c / 'v3/singlepass_metrics.csv')
    results = []
    for model, expected in counts.iterrows():
        frame = load_scores(ROOT, model)
        assert frame.sample_id.is_unique
        cal = frame[frame.split == 'calib']
        test = frame[frame.split == 'test_website']
        assert len(test) == expected.n
        assert int(((test.iou_fail == 1) & (test.click_fail == 0)).sum()) == expected.iou_fail_point_hit
        assert int(((test.iou_fail == 0) & (test.click_fail == 1)).sum()) == expected.iou_ok_point_fail == 0
        score = failure_score(test, 'geom_token_logistic')
        acc = score <= np.median(failure_score(cal, 'geom_token_logistic'))
        ids = set(accepted.loc[accepted.model == model, 'sample_id'])
        assert set(test.loc[acc, 'sample_id']) == ids
        assert int(acc.sum()) == risks.loc[model, 'n_accepted']
        assert round(float(test.loc[acc, 'click_fail'].mean()), 4) == risks.loc[model, 'o2_point_fail_rate']
        assert round(float(test.loc[acc, 'iou_fail'].mean()), 4) == risks.loc[model, 'o1_iou_fail_rate']
        auc = roc_auc_score(test.click_fail, score)
        ref = singles[(singles.model == model) & (singles.population == 'all852') & (singles['head'] == 'geom_token_logistic')].iloc[0]
        assert abs(auc - ref.auroc) < 1e-13
        raw = failure_score(test, 'raw_coord_mean_entropy') <= np.median(failure_score(cal, 'raw_coord_mean_entropy'))
        op = points[(points.model == model) & (points.population == 'test_website')].iloc[0]
        assert int(raw.sum()) == op.n_accepted
        assert int(test.loc[raw, 'click_fail'].sum()) == op.n_accepted_failures
        assert abs(float(raw.mean()) - op.coverage) < 1e-13
        assert abs(float(test.loc[raw, 'click_fail'].mean()) - op.accepted_risk) < 1e-13
        results.append({'model': model, 'n': len(test), 'discordant': int(expected.iou_fail_point_hit),
                        'geometry_token_accepted': int(acc.sum()), 'geometry_token_auroc': float(auc),
                        'raw_entropy_coverage': float(raw.mean()), 'raw_entropy_risk': float(op.accepted_risk)})
    delta = pd.read_csv(c / 'aggregate_audit/observed_and_bootstrap_macro_deltas.csv')
    primary = delta[delta.contrast == 'geom_token_logistic-token_logistic'].iloc[0]
    assert primary.paired_ci_lo < 0 < primary.paired_ci_hi
    print(json.dumps({'status': 'PASS', 'scope': 'inventory and principal numeric smoke only; not full L1/L2',
                      'inventory_files': count, 'web_results': results,
                      'primary_increment_interval_includes_zero': True}, indent=2))


if __name__ == '__main__':
    main()
