#!/usr/bin/env python3
"""Independent numeric/semantic checks for V3; optional --replay compares fresh outputs.

python src/scripts_rev/v3_semantic_checks.py . --replay replay_v3
"""
import argparse
import sys
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parent))
from entropy_scale import load_scores


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--replay',type=Path)
    a=ap.parse_args();root=a.root;d=root/'data_compact/corrections/entropy_scale_v1/v3';checks=[]
    def check(name,ok):
        checks.append(bool(ok));print(('PASS ' if ok else 'FAIL ')+name)
    frozen=json.loads((root/'protocol/V3_FROZEN_INPUT_SHA256.json').read_text())
    check('all public input pins verify; privacy-minimized manifest documented separately',
          all(hashlib.sha256((root/f).read_bytes()).hexdigest()==sha for f,sha in frozen.items()))
    # Independent exact-integer geometry check, including target-boundary equality.
    ranges=list(itertools.combinations(range(-2,3),2))
    boxes=[(x1,y1,x2,y2) for (x1,x2),(y1,y2) in itertools.product(ranges,ranges)]
    invalid=0; tested=0; boundaries=0
    for p,g in itertools.product(boxes,repeat=2):
        inter=max(0,min(p[2],g[2])-max(p[0],g[0]))*max(0,min(p[3],g[3])-max(p[1],g[1]))
        area_p=(p[2]-p[0])*(p[3]-p[1]);area_g=(g[2]-g[0])*(g[3]-g[1])
        o1=2*inter >= area_p+area_g-inter
        o2=2*g[0]<=p[0]+p[2]<=2*g[2] and 2*g[1]<=p[1]+p[3]<=2*g[3]
        invalid+=int(o1 and not o2);tested+=1
        boundaries+=int(o1 and (p[0]+p[2] in [2*g[0],2*g[2]] or p[1]+p[3] in [2*g[1],2*g[3]]))
    check(f'exact rectangle implication ({tested} pairs; {boundaries} boundary successes)',invalid==0 and boundaries>0)
    # No epsilon is introduced into frozen labels. Distinguish small float perturbations.
    for delta in [-1e-8,0,1e-8]:
        inter=1-delta;union=2+delta;iou=inter/union;center_in=(1 >= 1+delta)
        check(f'inclusive boundary with delta={delta:g}',not(iou>=.5 and not center_in))
    w=pd.read_parquet(root/'data_compact/e3_web/per_row_web.parquet')
    cal=pd.read_csv(d/'probability_calibration.csv')
    q=pd.read_csv(d/'queue_metrics.csv');pair=pd.read_csv(d/'paired_queue_contrasts.csv')
    ids=pd.read_csv(d/'queue_ids.csv');scores=pd.read_csv(d/'exploratory_scores.csv')
    single=pd.read_csv(d/'singlepass_metrics.csv')
    for m,expected in [('qwen3vl4b',(188,185)),('internvl35_4b',(541,148))]:
        ss=load_scores(root,m)
        tw=ss[ss.split=='test_website'];fit=w[(w.model==m)&(w.split=='fit')]
        check(m+' population: 852 all and 670 CLICK',len(tw)==852 and (tw.op=='CLICK').sum()==670)
        check(m+' nested labels / measured disagreement',not((tw.iou_fail==0)&(tw.click_fail==1)).any()
              and int(tw.click_fail.sum())==expected[0]
              and int(((tw.iou_fail==1)&(tw.click_fail==0)).sum())==expected[1])
        c=ss[ss.split=='calib'];tau=np.median(1-c.prob_success_geom_token_logistic)
        acc=(1-tw.prob_success_geom_token_logistic)<=tau
        gap=tw.loc[acc,'iou_fail'].mean()-tw.loc[acc,'click_fail'].mean()
        discord=((tw.loc[acc,'iou_fail']==1)&(tw.loc[acc,'click_fail']==0)).mean()
        check(m+' common-set risk-gap identity',abs(gap-discord)<1e-14)
        sr=single[(single.model==m)&(single['head']=='geom_token_logistic')&(single.population=='all852')].iloc[0]
        check(m+' original median threshold unchanged',abs(sr.threshold-tau)<1e-14)
        for split in ['test_website','test_domain']:
            t=ss[ss.split==split];success=1-t.click_fail.to_numpy();pf=float((1-fit.click_fail).mean())
            for _,r in cal[(cal.model==m)&(cal.split==split)].iterrows():
                p=t['prob_success_'+r['head']].to_numpy()
                check(m+' '+split+' '+r['head']+' Brier against sklearn',abs(r.brier-brier_score_loss(success,p))<1e-13)
                check(m+' '+split+' fit-only constant / separate hindsight oracle',
                      abs(r.fit_success_rate-pf)<1e-14 and abs(r.brier_fit_constant-np.mean((pf-success)**2))<1e-13
                      and abs(r.brier_test_oracle-np.var(success))<1e-13
                      and abs(r.brier_fit_constant-r.brier_test_oracle-(pf-success.mean())**2)<1e-13)
                # Independent bin loop with explicit boundaries, not production searchsorted.
                e=0
                for b in range(10):
                    ix=(p>=0)&(p<=.1) if b==0 else (p>b/10)&(p<=(b+1)/10)
                    if ix.any():e+=ix.mean()*abs(p[ix].mean()-success[ix].mean())
                check(m+' '+split+' '+r['head']+' ECE ten bins',abs(e-r.ece10)<1e-13)
        s=scores[(scores.model==m)&(scores.split=='test_website')].set_index('sample_id')
        # Independent ID-sorted ranking; reverse input order must not affect selected IDs.
        for frac in [.1,.2,.3]:
            k=math.ceil(frac*len(s))
            for h in ['geom_token_O1','geom_token_O2','token_O1','token_O2']:
                def independent_ids(frame):
                    return set(sorted(frame.index,key=lambda sid:(-frame.loc[sid,h],
                          hashlib.sha256(('VLMGF-V3-20261021|'+sid).encode()).hexdigest()))[:k])
                expect=independent_ids(s)
                got=set(ids[(ids.model==m)&(ids.population=='all852')&(ids.fraction==frac)&(ids.score==h)].sample_id)
                check(f'{m} {h} {frac} queue IDs and permutation invariance',got==expect==independent_ids(s.iloc[::-1]))
                r=q[(q.model==m)&(q.population=='all852')&(q.fraction==frac)&(q.score==h)].iloc[0]
                queue=s.loc[list(got)]
                check(f'{m} {h} {frac} caught/discord accounting',r.queue_size==k and r.caught==queue.click_fail.sum()
                      and r.discord_count==((queue.iou_fail==1)&(queue.click_fail==0)).sum()
                      and abs(r.discord_fraction-r.discord_count/k)<1e-14)
    check('primary and CLICK no resolved O2-fit geometry+token caught gain',
          (pair.query("score_a=='geom_token_O2' and score_b=='geom_token_O1' and population!='domain504_secondary'").delta_caught_lo<=0).all()
          and (pair.query("score_a=='geom_token_O2' and score_b=='geom_token_O1' and population!='domain504_secondary'").delta_caught_hi>=0).all())
    for m,path in [('qwen3vl4b','r9_1'),('internvl35_4b','r9_1/internvl35_4b')]:
        s=json.loads((root/('data_compact/corrections/entropy_scale_v1/r9_'+m)/'eval/r9_summary.json').read_text())
        c=s['deltas']['sg_code_default-minus-token_only'];p=s['deltas']['sg_paper_params-minus-token_only']
        check(m+' matched500 code-default resolved; published parameters unresolved',s['n_test']==500 and c['lo']*c['hi']>0 and p['lo']<=0<=p['hi'])
    coeff=pd.read_csv(root/'data_compact/corrections/entropy_scale_v1/p0/p1a_factor_terms_fdr.csv')
    check('factor semantics: 11 terms, 9 gate stable, 7 Bonferroni survivors',len(coeff)==11
          and coeff.stable_rev_gate.sum()==9 and (coeff.stable_rev_gate&coeff.bonferroni_sig_05).sum()==7)
    claims=json.loads((root/'protocol/V3_SEMANTIC_EXPECTATIONS.json').read_text())
    check('labels vs scored inputs / O2 proxy / pointwise / sampling qualifier',
          claims['threshold_outcome_labels_required']==0 and claims['head_fit_labels']==510 and claims['threshold_scored_inputs']==200
          and claims['O2']=='static point-in-target localization proxy' and claims['bands']=='pointwise conditional on frozen heads'
          and claims['sampling_headline']=='released-code defaults versus fitted token-only on matched500; paper intervals include zero')
    if a.replay:
        for f in sorted(d.glob('*.csv')):
            try:assert_frame_equal(pd.read_csv(f),pd.read_csv(a.replay/f.name),check_exact=False,atol=1e-12,rtol=1e-12);ok=True
            except (AssertionError,FileNotFoundError) as e:print(str(e)[:300]);ok=False
            check('fresh outgoing-artifact replay '+f.name,ok)
    else:
        print('SKIP fresh-output comparison (supply --replay); independent checks above still executed')
    print(f'{sum(checks)}/{len(checks)} semantic/numeric checks PASS; {len(checks)-sum(checks)} FAIL')
    raise SystemExit(0 if all(checks) else 1)


if __name__=='__main__':main()
