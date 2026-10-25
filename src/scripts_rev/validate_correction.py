"""Changed/unchanged result checks against immutable legacy data, not self-consistency alone."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from entropy_scale import correction_root,load_scores,RAW

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):
    a=a.reset_index(drop=True).copy();b=b.reset_index(drop=True).copy()
    # A changed raw-entropy NA can change a CSV column's inferred dtype even
    # when its fitted-head rows are identical. Compare numeric values as numbers.
    for c in a.columns:
        try:aa=pd.to_numeric(a[c]);bb=pd.to_numeric(b[c])
        except (TypeError,ValueError):continue
        a[c]=aa;b[c]=bb
    assert_frame_equal(a,b,check_exact=False,check_dtype=False,rtol=1e-13,atol=1e-14)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--out-dir',type=Path)
    a=p.parse_args();root=a.root;out=correction_root(root);checks=[];changed=[]
    def passed(name):checks.append({'check':name,'pass':True});print('PASS',name,flush=True)
    new=pd.read_csv(out/'corrected_raw_entropy_operating_points.csv').sort_values(['model','population'])
    if a.reference:
        ref=pd.read_csv(a.reference/'corrected_raw_entropy_operating_points.csv').sort_values(['model','population'])
        equal(new[ref.columns],ref);passed('all ten independent pyarrow-reference operating points agree to 1e-12')
        scores=pd.read_csv(out/'corrected_raw_entropy_scores.csv').sort_values(['model','split','sample_id'])
        refs=pd.read_csv(a.reference/'corrected_raw_entropy_scores.csv').sort_values(['model','split','sample_id'])
        equal(scores[refs.columns],refs);passed('all independent reference per-ID scores/acceptance agree')
    for name in ['exploratory_scores.csv','probability_calibration.csv','refit_agreement.csv',
                 'risk_coverage_pointwise.csv','queue_metrics.csv','paired_queue_contrasts.csv','queue_ids.csv']:
        equal(pd.read_csv(out/'v3'/name),pd.read_csv(root/'data_compact/v3'/name))
        passed('unchanged V3 '+name)
    for name in ['singlepass_metrics.csv','site_quantile_sensitivity.csv']:
        n=pd.read_csv(out/'v3'/name);old=pd.read_csv(root/'data_compact/v3'/name)
        equal(n[n['head']!='raw_coord_mean_entropy'],old[old['head']!='raw_coord_mean_entropy'])
        if name=='singlepass_metrics.csv':
            keep=['model','population','head','n','n_fail','caught20','k20','auroc']
            equal(n[keep],old[keep])
        passed('only raw-entropy operating-point fields change: '+name)
    for model,prefix in [('qwen3vl4b','r9_1'),('internvl35_4b','r9_1/internvl35_4b')]:
        n=json.loads((out/('r9_'+model)/'eval/r9_summary.json').read_text())
        old=json.loads((root/prefix/'eval/r9_summary.json').read_text())
        assert n['point_metrics']==old['point_metrics']
        assert n['deltas']==old['deltas'] and n['primary']==old['primary']
        for key in old['frozen_threshold_50pct_calibration']:
            if key!='raw_coord_entropy':assert n['frozen_threshold_50pct_calibration'][key]==old['frozen_threshold_50pct_calibration'][key]
        passed(model+' sampling point metrics, paired AUROC effects/intervals and all non-entropy thresholds unchanged')
        sc=load_scores(root,model);legacy=pd.read_parquet(root/f'data_compact/final_audit/per_row_scores_{model}.parquet')
        cal=sc[sc.split=='calib'];tw=sc[sc.split=='test_website']
        tau=np.median(1-cal.prob_success_geom_token_logistic)
        accepted=tw[(1-tw.prob_success_geom_token_logistic)<=tau]
        expected=pd.read_csv(root/'data_compact/p0_2026-10-20/p0b_accepted_ids_geom_token_frozen.csv')
        assert set(accepted.sample_id)==set(expected[expected.model==model].sample_id)
        assert len(accepted)==(457 if model=='qwen3vl4b' else 322)
        for col in legacy:
            if col!='prob_success_raw_coord_mean_entropy':pd.testing.assert_series_equal(sc[col],legacy[col])
        passed(model+' every fitted score/label/split/group and geometry+token threshold/accepted ID unchanged')
        oldtau=np.median(1-legacy.loc[legacy.split=='calib','prob_success_raw_coord_mean_entropy'])
        for split in ['calib','test_website','test_domain']:
            before=legacy[legacy.split==split];after=sc[sc.split==split]
            b=(1-before.prob_success_raw_coord_mean_entropy)<=oldtau
            now=after.accepted_frozen_raw_entropy
            changed.append(dict(model=model,split=split,old_threshold_normalized=float(oldtau),
                old_accepted=int(b.sum()),old_coverage=float(b.mean()),old_risk=float(before.loc[b,'click_fail'].mean()),
                corrected_accepted=int(now.sum()),corrected_coverage=float(now.mean()),corrected_risk=float(after.loc[now,'click_fail'].mean()),
                changed_ids=sorted(set(before.loc[b,'sample_id'])^set(after.loc[now,'sample_id']))))
    for name in ['p0b_same_accepted_set_two_outcomes.csv','p0b_accepted_ids_geom_token_frozen.csv','p0c_risk_coverage_bands.csv','p0b_outcome_contingency_852.csv']:
        equal(pd.read_csv(out/'p0'/name),pd.read_csv(root/'data_compact/p0_2026-10-20'/name));passed('unchanged '+name)
    for name in ['p0c_heads_overall_and_domain.csv','p0b_heads_click_only_670.csv','p0d_by_action_type_frozen.csv',
                 'p0d_by_size_tercile_frozen.csv','p0d_model_by_website_frozen.csv','p0d_leave_one_website_out.csv']:
        n=pd.read_csv(out/'p0'/name);old=pd.read_csv(root/'data_compact/p0_2026-10-20'/name)
        equal(n[n['head']!='raw_coord_mean_entropy'],old[old['head']!='raw_coord_mean_entropy'])
        passed('non-entropy rows unchanged in '+name)
    frozen=json.loads((root/'protocol/V3_FROZEN_INPUT_SHA256.json').read_text())
    assert all(sha(root/n)==h for n,h in frozen.items());passed(str(len(frozen))+' published input hashes match the public pin manifest; privacy changes documented separately')
    fig=root/'figure_sources/fig1_refined/fig1_refined_candidate.pdf'
    if fig.exists():
        assert sha(fig)=='65d416c8a831a14ddee2d7020c27060b5516f80df9704fc22880c83b50f7567e';passed('real Fig1Refined PDF bytes unchanged')
    else:
        print('EXCLUDED: Figure 1 screenshot not redistributed; not counted as a passed asset check.')
    report=a.out_dir or out
    report.mkdir(parents=True,exist_ok=True)
    (report/'changed_acceptance_ids.json').write_text(json.dumps(changed,indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in row.items() if k!='changed_ids'}|{'changed_ids':len(row['changed_ids'])} for row in changed]).to_csv(report/'old_new_operating_points.csv',index=False)
    (report/'correction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(len(checks),'changed/unchanged checks PASS')

if __name__=='__main__':main()
