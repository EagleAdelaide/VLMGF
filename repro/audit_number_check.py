"""Portable checks of current corrected manuscript quantities, with named comparators."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'data_compact/corrections/entropy_scale_v1'
d=pd.read_csv(C/'corrected_raw_entropy_operating_points.csv')
expected={('qwen3vl4b','test_website'):(375,45,.44,.12),
 ('internvl35_4b','test_website'):(314,109,.369,.347),
 ('qwen3vl4b','sampling500'):(204,25,.408,.123),
 ('internvl35_4b','sampling500'):(165,59,.330,.358)}
for (model,pop),(n,nfail,cov,risk) in expected.items():
    r=d[(d.model==model)&(d.population==pop)].iloc[0]
    assert r.n_accepted==n and r.n_accepted_failures==nfail
    assert round(r.coverage,3)==round(cov,3) and round(r.accepted_risk,3)==round(risk,3)
    print('PASS corrected entropy',model,pop,n,nfail)
n=pd.read_csv(C/'aggregate_audit/natural_image_model_macro_baselines.csv').set_index('method')
assert round(n.loc['full_gbdt','risk_at_50'],3)==.155
assert round(n.loc['geom_gbdt','risk_at_50'],3)==.180
assert round(n.loc['full_gbdt','prevalence'],3)==.284
print('PASS named natural-image comparator: full .155 / geometry .180 / accept-all .284')
a=pd.read_csv(C/'aggregate_audit/observed_and_bootstrap_macro_deltas.csv')
r=a[a.contrast=='geom_token_logistic-token_logistic'].iloc[0]
assert round(r.observed_model_macro_delta,3)==.013
assert round(r.bootstrap_mean,3)==.012
assert round(r.paired_ci_lo,3)==-.005 and round(r.paired_ci_hi,3)==.030
print('PASS observed model-macro delta .013, separate bootstrap mean .012, original paired interval retained')
v=pd.read_csv(C/'p0/p0b_same_accepted_set_two_outcomes.csv').set_index('model')
assert v.loc['qwen3vl4b','n_accepted']==457 and v.loc['internvl35_4b','n_accepted']==322
print('PASS Figure 1 accepted sets 457/322')
