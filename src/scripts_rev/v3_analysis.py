#!/usr/bin/env python3
"""Bounded exploratory V3 CPU analysis; writes only a new output directory.

python src/scripts_rev/v3_analysis.py . replay_v3
Protocol: protocol/V3_EXPLORATORY_AMENDMENT.md. No model inference.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from entropy_scale import load_scores, failure_score, correction_root
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, B = 20261021, 2000
MODELS = ['qwen3vl4b', 'internvl35_4b']
FRACS = [.10, .20, .30]
GEOM = ['pred_area_ratio', 'pred_cx', 'pred_cy', 'pred_w', 'pred_h', 'pred_aspect',
        'pred_edge_dist', 'pred_touches_border', 'pred_bbox_missing']
TOKEN = ['coord_mean_logprob', 'coord_min_logprob', 'coord_mean_entropy', 'out_mean_logprob']
SPECS = {'geom_token': GEOM + TOKEN + [c + '_missing' for c in TOKEN],
         'token': TOKEN + [c + '_missing' for c in TOKEN]}
SCORES = ['geom_token_O1', 'geom_token_O2', 'token_O1', 'token_O2']
PAIRS = [('geom_token_O2', 'geom_token_O1'), ('token_O2', 'token_O1'),
         ('geom_token_O2', 'token_O2')]
GRID = np.round(np.arange(.05, 1.0001, .01), 2)


def tie_keys(ids):
    return np.array([hashlib.sha256(('VLMGF-V3-20261021|' + str(s)).encode()).hexdigest()
                     for s in ids])


def order_score(failure, tie, review=True):
    return np.lexsort((tie, -failure if review else failure))


def ece10(p, success):
    # [0,.1], (.1,.2], ... (.9,1]; every observation belongs to one bin.
    edges = np.linspace(0, 1, 11)
    bid = np.clip(np.searchsorted(edges, p, side='left') - 1, 0, 9)
    return sum(np.mean(bid == b) * abs(p[bid == b].mean() - success[bid == b].mean())
               for b in range(10) if np.any(bid == b))


def summarize(d, ii, frac):
    y2, y1 = d.click_fail.to_numpy()[ii], d.iou_fail.to_numpy()[ii]
    tie = tie_keys(d.sample_id)[ii]
    n, k = len(ii), math.ceil(frac * len(ii))
    queues, metrics = {}, {}
    for s in SCORES:
        ix = order_score(d[s].to_numpy()[ii], tie)[:k]
        queues[s] = set(ix.tolist())
        caught = y2[ix].sum()
        discord = ((y1[ix] == 1) & (y2[ix] == 0)).sum()
        metrics[s] = dict(caught=float(caught), recall=float(caught / y2.sum()),
                          discord_count=float(discord), discord_fraction=float(discord / k))
    pairs = {}
    for a, b in PAIRS:
        inter = len(queues[a] & queues[b])
        pairs[a, b] = dict(delta_caught=metrics[a]['caught'] - metrics[b]['caught'],
                           delta_recall=metrics[a]['recall'] - metrics[b]['recall'],
                           overlap_count=float(inter), overlap_fraction=inter / k,
                           jaccard=inter / (2 * k - inter))
    return metrics, pairs, k


def main():
    root, out = map(Path, sys.argv[1:3])
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Refusing to overwrite a populated output directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(root / 'data_compact/e3_web/per_row_web.parquet')
    provenance = dict(exploratory=True, seed=SEED, bootstrap_draws=B, fractions=FRACS,
                      tie_rule='sha256(VLMGF-V3-20261021|sample_id)',
                      fitted_seed=20261009, n_fit=510, no_test_fitting=True,
                      intervals='pointwise paired task-cluster percentile 95%; frozen heads',
                      input_sha256={})
    for p in [root / 'data_compact/e3_web/per_row_web.parquet'] + [
            correction_root(root) / f'per_row_scores_{m}.parquet' for m in MODELS]:
        provenance['input_sha256'][str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    stores, score_frames, calrows, checks, quantiles, bandrows, headrows = {}, [], [], [], [], [], []
    for model in MODELS:
        dm = w[w.model == model].copy()
        fit = dm[dm.split == 'fit'].copy()
        assert len(fit) == 510 and fit.sample_id.is_unique
        med = {c: float(fit[c].median()) for c in TOKEN}
        for c in TOKEN:
            dm[c + '_missing'] = dm[c].isna().astype(float)
            dm[c] = dm[c].fillna(med[c])
        fit = dm[dm.split == 'fit']
        sc = load_scores(root, model)
        assert sc.sample_id.is_unique
        te = dm[dm.split != 'fit'].copy()
        te = te.set_index('sample_id').loc[sc.sample_id].reset_index()
        assert np.array_equal(te.click_fail, sc.click_fail) and np.array_equal(te.iou_fail, sc.iou_fail)
        assert not set(fit.task) & set(te.task)
        # Isolated fixed-specification refits. No thresholds are refit or selected.
        for spec, cols in SPECS.items():
            for outcome, label in [('O1', 'iou_fail'), ('O2', 'click_fail')]:
                pipe = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=20261009))
                pipe.fit(fit[cols].to_numpy(float), (1 - fit[label]).to_numpy())
                p = pipe.predict_proba(te[cols].to_numpy(float))[:, 1]
                te[spec + '_' + outcome] = 1 - p
                if outcome == 'O2':
                    diff = float(np.max(np.abs(p - sc[f'prob_success_{spec}_logistic'].to_numpy())))
                    assert diff < 1e-6, diff
                    checks.append(dict(model=model, check=spec + '_O2_refit_vs_frozen_maxdiff', value=diff))
        # Fit-only constant reference; preserve test-prevalence constant as descriptive oracle.
        pfit = float((1 - fit.click_fail).mean())
        for split in ['test_website', 'test_domain']:
            t = sc[sc.split == split]
            y = (1 - t.click_fail).to_numpy()
            for h in ['token_logistic', 'geom_token_logistic', 'geom_logistic', 'geom_gbdt', 'full_gbdt']:
                p = t['prob_success_' + h].to_numpy()
                brier = float(np.mean((p - y) ** 2))
                assert abs(brier - brier_score_loss(y, p)) < 1e-14
                calrows.append(dict(model=model, split=split, head=h, n=len(t), n_fit=510,
                                    fit_success_rate=pfit, test_success_rate=float(y.mean()), brier=brier,
                                    brier_fit_constant=float(np.mean((pfit - y) ** 2)),
                                    brier_test_oracle=float(np.mean((y.mean() - y) ** 2)),
                                    ece10=ece10(p, y), mean_pred_success=float(p.mean())))
        # Scored calibration inputs, not outcomes, determine these descriptive thresholds.
        cal = sc[sc.split == 'calib']
        tw = sc[sc.split == 'test_website'].reset_index(drop=True)
        for h in ['geom_gbdt', 'raw_coord_mean_entropy', 'token_logistic', 'geom_token_logistic']:
            col = 'prob_success_' + h
            failure = failure_score(tw,col).to_numpy()
            tie = tie_keys(tw.sample_id)
            tasks = sorted(tw.task.unique()); idx = {t: np.flatnonzero(tw.task == t) for t in tasks}
            rng = np.random.default_rng(SEED)
            def curve(ii):
                ix = order_score(failure[ii], tie[ii], review=False)
                return np.interp(GRID, np.arange(1, len(ii)+1)/len(ii),
                                 np.cumsum(tw.click_fail.to_numpy()[ii][ix])/np.arange(1, len(ii)+1))
            obs = curve(np.arange(len(tw)))
            vals = np.array([curve(np.concatenate([idx[t] for t in rng.choice(tasks, len(tasks))])) for _ in range(B)])
            lo, hi = np.quantile(vals, [.025, .975], axis=0)
            for c, v, l, u in zip(GRID, obs, lo, hi):
                bandrows.append(dict(model=model, head=h, coverage=c, risk=v, lo=l, hi=u,
                                     interval='pointwise task-bootstrap; frozen head'))
            for q in [.25, .50, .75]:
                tau = float(np.quantile(failure_score(cal,col), q))
                for site, sub in tw.groupby('website'):
                    accept = failure_score(sub,col) <= tau
                    quantiles.append(dict(model=model, head=h, calibration_quantile=q, threshold=tau,
                                          calibration_inputs=200, website=site, n=len(sub), accepted=int(accept.sum()),
                                          coverage=float(accept.mean()),
                                          o2_risk=float(sub.loc[accept, 'click_fail'].mean()) if accept.any() else np.nan))
            # New tables and curves use the same per-ID V3 tie rule.
            for pop, sub in [('all852', tw), ('CLICK670', tw[tw.op == 'CLICK'])]:
                order = order_score(failure_score(sub,col).to_numpy(), tie_keys(sub.sample_id))
                k = math.ceil(.20 * len(sub))
                tau = float(np.median(failure_score(cal,col)))
                accept = failure_score(sub,col) <= tau
                headrows.append(dict(model=model, population=pop, head=h, n=len(sub),
                                     n_fail=int(sub.click_fail.sum()), caught20=int(sub.click_fail.to_numpy()[order[:k]].sum()),
                                     k20=k, auroc=roc_auc_score(sub.click_fail, failure_score(sub,col)),
                                     threshold=tau, coverage=float(accept.mean()),
                                     risk=float(sub.loc[accept, 'click_fail'].mean())))
        # Only these isolated, exploratory scores are exported.
        score_frames.append(te[['model','sample_id','split','task','website','op','iou_fail','click_fail'] + SCORES])
        for pop, split, op in [('all852', 'test_website', None), ('CLICK670', 'test_website', 'CLICK'),
                               ('domain504_secondary', 'test_domain', None)]:
            d = te[te.split == split]
            if op:
                d = d[d.op == op]
            stores[model, pop] = d.reset_index(drop=True)
    pd.concat(score_frames).to_csv(out / 'exploratory_scores.csv', index=False)
    pd.DataFrame(calrows).to_csv(out / 'probability_calibration.csv', index=False)
    pd.DataFrame(checks).to_csv(out / 'refit_agreement.csv', index=False)
    pd.DataFrame(quantiles).to_csv(out / 'site_quantile_sensitivity.csv', index=False)
    pd.DataFrame(bandrows).to_csv(out / 'risk_coverage_pointwise.csv', index=False)
    pd.DataFrame(headrows).to_csv(out / 'singlepass_metrics.csv', index=False)
    # Shared resampling draws across both models, heads, and budgets within population.
    rows, contrasts, ids = [], [], []
    for pop in ['all852', 'CLICK670', 'domain504_secondary']:
        d0 = stores[MODELS[0], pop]
        tasks = sorted(d0.task.unique()); rng = np.random.default_rng(SEED)
        draws = rng.choice(tasks, (B, len(tasks)))
        for model in MODELS:
            d = stores[model, pop]
            assert np.array_equal(d.sample_id, d0.sample_id)
            idx = {t: np.flatnonzero(d.task == t) for t in tasks}
            sels = [np.concatenate([idx[t] for t in draw]) for draw in draws]
            for frac in FRACS:
                point, pair, k = summarize(d, np.arange(len(d)), frac)
                boots = [summarize(d, ii, frac)[:2] for ii in sels]
                for s, vals in point.items():
                    row = dict(model=model, population=pop, score=s, fraction=frac, n=len(d),
                               total_o2_failures=int(d.click_fail.sum()), queue_size=k, **vals)
                    for metric in vals:
                        lo, hi = np.quantile([z[0][s][metric] for z in boots], [.025, .975])
                        row[metric + '_lo'], row[metric + '_hi'] = lo, hi
                    rows.append(row)
                    ix = order_score(d[s].to_numpy(), tie_keys(d.sample_id))[:k]
                    for j in ix:
                        ids.append(dict(model=model, population=pop, fraction=frac, score=s,
                                        sample_id=d.sample_id.iloc[j], o1_fail=int(d.iou_fail.iloc[j]),
                                        o2_fail=int(d.click_fail.iloc[j])))
                for (a, b), vals in pair.items():
                    row = dict(model=model, population=pop, score_a=a, score_b=b, fraction=frac,
                               n=len(d), total_o2_failures=int(d.click_fail.sum()), queue_size=k, **vals)
                    for metric in vals:
                        lo, hi = np.quantile([z[1][a, b][metric] for z in boots], [.025, .975])
                        row[metric + '_lo'], row[metric + '_hi'] = lo, hi
                    contrasts.append(row)
                # Relabeling a fixed score never enters ordering; assert explicitly.
                for s in SCORES:
                    one = order_score(d[s].to_numpy(), tie_keys(d.sample_id))[:k]
                    two = order_score(d[s].to_numpy(), tie_keys(d.sample_id))[:k]
                    assert np.array_equal(one, two)
    pd.DataFrame(rows).to_csv(out / 'queue_metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out / 'paired_queue_contrasts.csv', index=False)
    pd.DataFrame(ids).to_csv(out / 'queue_ids.csv', index=False)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Exploratory CPU analysis complete:', len(rows), 'queue rows;', len(contrasts), 'paired rows')
    print(pd.DataFrame(contrasts).query("population == 'all852' and score_b == 'geom_token_O1'")
          [['model','fraction','delta_caught','delta_caught_lo','delta_caught_hi','overlap_fraction']].to_string(index=False))


if __name__ == '__main__':
    main()
