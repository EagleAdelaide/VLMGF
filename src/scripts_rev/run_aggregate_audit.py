"""Observed model-macro effects, original paired intervals, and named risk baselines."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from entropy_scale import correction_root,load_scores

def auc(score,y):
    rank=pd.Series(score).rank(method='average').to_numpy();pos=y==1;n1=int(pos.sum());n0=int((~pos).sum())
    return (rank[pos].sum()-n1*(n1+1)/2)/(n1*n0)

def main():
    root,out=map(Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(root/'data_compact/e1a_point_metrics.csv')
    natural=d.groupby('method',sort=True).mean(numeric_only=True)
    natural.to_csv(out/'natural_image_model_macro_baselines.csv')
    assert abs(natural.loc['geom_gbdt','risk_at_50']-.18033333333333335)<1e-12
    assert abs(natural.loc['full_gbdt','risk_at_50']-.15544444444444444)<1e-12
    assert abs(natural.loc['full_gbdt','prevalence']-.2842222222222222)<1e-12
    old=json.loads((root/'data_compact/final_audit/WEB_DELTAS_BOOTSTRAP.json').read_text())
    models=['qwen3vl4b','internvl35_4b'];frames={m:load_scores(root,m).query("split=='test_website'").reset_index(drop=True) for m in models}
    tasks=sorted(set.union(*[set(f.task) for f in frames.values()]))
    groups={m:[np.flatnonzero(f.task==t) for t in tasks] for m,f in frames.items()}
    scores={m:{h:1-f['prob_success_'+h].to_numpy() for key in old for h in key.split('-')} for m,f in frames.items()}
    ys={m:f.click_fail.to_numpy() for m,f in frames.items()}
    observed={k:float(np.mean([auc(scores[m][k.split('-')[0]],ys[m])-auc(scores[m][k.split('-')[1]],ys[m]) for m in models])) for k in old}
    draws={k:[] for k in old};rng=np.random.default_rng(20261009)
    for b in range(2000):
        dr=rng.integers(0,len(tasks),len(tasks))
        ix={m:np.concatenate([groups[m][g] for g in dr]) for m in models}
        for key in old:
            a,c=key.split('-')
            draws[key].append(float(np.mean([auc(scores[m][a][ix[m]],ys[m][ix[m]])-auc(scores[m][c][ix[m]],ys[m][ix[m]]) for m in models])))
        if (b+1)%500==0:print('paired task-cluster aggregate draws',b+1,flush=True)
    rows=[]
    for key,values in draws.items():
        mean=float(np.mean(values));lo,hi=np.quantile(values,[.025,.975])
        assert abs(mean-old[key]['mean'])<1e-12,(key,mean,old[key])
        assert abs(lo-old[key]['lo'])<1e-12 and abs(hi-old[key]['hi'])<1e-12,key
        rows.append(dict(contrast=key,observed_model_macro_delta=observed[key],bootstrap_mean=mean,
                         paired_ci_lo=lo,paired_ci_hi=hi,models=2,task_clusters=len(tasks),draws=2000))
    pd.DataFrame(rows).to_csv(out/'observed_and_bootstrap_macro_deltas.csv',index=False)
    (out/'aggregate_reporting.json').write_text(json.dumps({'averaging_unit':'equal-weight mean of the two model-specific AUROC differences',
        'point_estimate':'observed full-sample statistic; bootstrap mean separately identified',
        'interval':'original synchronized task-cluster percentile 95%, 2000 draws, seed 20261009; fixed heads',
        'all_original_intervals_and_bootstrap_means_reproduced':True},indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
