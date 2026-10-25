"""Reconstruct a deterministic corrected score layer from immutable saved features."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parent))
from entropy_scale import EntropyRule, RAW, correction_root

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path)
    a=p.parse_args();root=a.artifact_root;out=a.out_dir or correction_root(root)
    out.mkdir(parents=True,exist_ok=True)
    source=root/'data_compact/e3_web/per_row_web.parquet'
    raw=pd.read_parquet(source);metrics=[];perrows=[];rules={};checks=[]
    for model in ['qwen3vl4b','internvl35_4b']:
        dm=raw[raw.model==model].set_index('sample_id',drop=False)
        fit=dm[dm.split=='fit'];cal=dm[dm.split=='calib']
        assert len(fit)==510 and len(cal)==200
        rule=EntropyRule.from_development(fit.coord_mean_entropy,cal.coord_mean_entropy)
        oldpath=root/f'data_compact/final_audit/per_row_scores_{model}.parquet'
        legacy=pd.read_parquet(oldpath);aligned=dm.loc[legacy.sample_id]
        assert np.array_equal(aligned.split,legacy.split)
        assert np.array_equal(aligned.click_fail,legacy.click_fail)
        assert np.array_equal(aligned.iou_fail,legacy.iou_fail)
        corrected=legacy.copy()
        corrected=corrected.rename(columns={'prob_success_raw_coord_mean_entropy':'legacy_split_normalized_entropy_success_like_score'})
        corrected[RAW]=rule.score(aligned.coord_mean_entropy)
        corrected['accepted_frozen_raw_entropy']=rule.accept(aligned.coord_mean_entropy)
        corrected.to_parquet(out/f'per_row_scores_{model}.parquet',index=False)
        rules[model]={'fit_imputation_median':rule.imputation_median,'calibration_threshold':rule.threshold,
            'n_fit':510,'n_calibration':200,'accept':'raw_entropy_failure_score <= calibration_threshold',
            'score_units':'coordinate-token entropy, natural log units; not a probability',
            'legacy_score_sha256':sha(oldpath),'raw_features_sha256':sha(source)}
        for c in legacy:
            if c=='prob_success_raw_coord_mean_entropy':continue
            pd.testing.assert_series_equal(legacy[c],corrected[c])
        tests=[('calib',corrected[corrected.split=='calib']),
               ('test_website',corrected[corrected.split=='test_website']),
               ('test_website_CLICK',corrected[(corrected.split=='test_website')&(corrected.op=='CLICK')]),
               ('test_domain',corrected[corrected.split=='test_domain'])]
        prefix='r9_1' if model=='qwen3vl4b' else 'r9_1/internvl35_4b'
        panel=pd.read_csv(root/prefix/'eval/r9_per_input_scores.csv').query("split=='test_website'")
        sample=corrected.set_index('sample_id').loc[panel.sample_id]
        assert len(sample)==500 and np.array_equal(sample.click_fail,panel.click_fail)
        tests.append(('sampling500',sample))
        for population,part in tests:
            accepted=part.accepted_frozen_raw_entropy.to_numpy(bool);y=part.click_fail.to_numpy(int)
            metrics.append(dict(model=model,population=population,n=len(part),n_fail=int(y.sum()),
                entropy_threshold=rule.threshold,n_accepted=int(accepted.sum()),
                n_accepted_failures=int(y[accepted].sum()),coverage=float(accepted.mean()),
                accepted_risk=float(y[accepted].mean()) if accepted.any() else None,
                auroc=float(roc_auc_score(y,part[RAW]))))
        for split in ['calib','test_website','test_domain']:
            part=corrected[corrected.split==split]
            risk=part[RAW].to_numpy();old=1-part.legacy_split_normalized_entropy_success_like_score.to_numpy()
            tie=np.array([hashlib.sha256(('VLMGF-V3-20261021|'+str(s)).encode()).hexdigest() for s in part.sample_id])
            for name,keys in [('V3_ID_hash',tie),('legacy_R9',np.random.default_rng(20261016).uniform(size=len(part))),('stable',np.arange(len(part)))]:
                for ascending in [True,False]:
                    first=np.lexsort((keys,risk if ascending else -risk));second=np.lexsort((keys,old if ascending else -old))
                    assert np.array_equal(first,second),(model,split,name)
                    y=part.click_fail.to_numpy();assert np.array_equal(np.cumsum(y[first]),np.cumsum(y[second]))
                checks.append({'model':model,'split':split,'check':name+' full rank order/curves/all rank budgets unchanged','pass':True})
            assert roc_auc_score(part.click_fail,risk)==roc_auc_score(part.click_fail,old)
            values=aligned[aligned.split==split].coord_mean_entropy.to_numpy()
            score=rule.score(values);accept=rule.accept(values)
            extras=np.r_[values,0.,np.max(score)+100.]
            assert np.array_equal(rule.score(extras)[:len(values)],score)
            assert np.array_equal(rule.accept(extras)[:len(values)],accept)
            order=np.random.default_rng(0).permutation(len(values));inverse=np.argsort(order)
            assert np.array_equal(rule.score(values[order])[inverse],score)
            assert np.array_equal(rule.accept(values[order])[inverse],accept)
            assert np.array_equal(np.concatenate([rule.score(c) for c in np.array_split(values,7)]),score)
            assert np.array_equal(np.concatenate([rule.accept(c) for c in np.array_split(values,7)]),accept)
            checks.append({'model':model,'split':split,'check':'unrelated extreme/permutation/chunk score and acceptance invariance','pass':True})
        per=corrected[['sample_id','split','click_fail',RAW,'accepted_frozen_raw_entropy']].copy();per['model']=model
        per['saved_coord_mean_entropy']=aligned.coord_mean_entropy.to_numpy()
        per['entropy_was_missing']=aligned.coord_mean_entropy.isna().to_numpy()
        perrows.append(per)
    # Explicit nontrivial missing-value and inclusive-tie regression cases.
    test=EntropyRule.from_development([1.,np.nan,3.],[1.,np.nan,3.])
    np.testing.assert_array_equal(test.score([np.nan,2.,3.]),[2.,2.,3.])
    np.testing.assert_array_equal(test.accept([np.nan,2.,3.]),[True,True,False])
    for values in [[np.inf],[np.nan,np.inf]]:
        try:test.score(values)
        except ValueError:pass
        else:raise AssertionError('Infinite entropy was silently accepted')
    checks.append({'check':'fit-only missing imputation, inclusive ties, and infinity rejection','pass':True})
    pd.DataFrame(metrics).to_csv(out/'corrected_raw_entropy_operating_points.csv',index=False)
    pd.concat(perrows,ignore_index=True).to_csv(out/'corrected_raw_entropy_scores.csv',index=False)
    (out/'frozen_entropy_parameters.json').write_text(json.dumps(rules,indent=2)+'\n')
    (out/'invariance_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False));print(len(checks),'rank/invariance checks PASS')

if __name__=='__main__':main()
