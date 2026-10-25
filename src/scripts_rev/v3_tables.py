"""Portable manuscript tables from frozen scores and isolated exploratory results."""
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from v3_analysis import tie_keys, order_score
from entropy_scale import load_scores, failure_score, correction_root

MODELS=['qwen3vl4b','internvl35_4b']
NAMES={'qwen3vl4b':'Qwen3-VL-4B','internvl35_4b':'InternVL3.5-4B'}


def write_table(path, spec, rows):
    path.write_text('\n'.join(['\\begin{tabular}{'+spec+'}', '\\toprule']+rows+
                              ['\\bottomrule','\\end{tabular}'])+'\n')


def tables(root,out):
    tab=out/'tables';tab.mkdir(parents=True,exist_ok=True)
    rows=[r'Score & Fit labels & \multicolumn{4}{c}{Qwen3-VL-4B} & \multicolumn{4}{c}{InternVL3.5-4B} \\',
          r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
          r' & & AUROC & C@20 & Coverage & Risk & AUROC & C@20 & Coverage & Risk \\',r'\midrule']
    frames={m:load_scores(root,m) for m in MODELS}
    specs=[('geom_gbdt','A: Geometry GBDT',510),('full_gbdt','A: Full GBDT',510),
           ('raw_coord_mean_entropy','B: Raw coord. entropy',0),('token_logistic','B: Token-only logistic',510),
           ('geom_token_logistic','B: Geom.+token logistic',510),('geom_token_gbdt','B: Geom.+token GBDT',510)]
    for h,name,labels in specs:
        cells=[name,str(labels)]
        for m in MODELS:
            s=frames[m];c=s[s.split=='calib'];t=s[s.split=='test_website'];click=t[t.op=='CLICK']
            col='prob_success_'+h
            auc=roc_auc_score(t.click_fail,failure_score(t,col));ac=roc_auc_score(click.click_fail,failure_score(click,col))
            ix=order_score(failure_score(t,col).to_numpy(),tie_keys(t.sample_id))[:math.ceil(.2*len(t))]
            caught=int(t.click_fail.to_numpy()[ix].sum());a=(failure_score(t,col))<=np.median(failure_score(c,col))
            cells += [f'{auc:.3f}',str(caught),f'{a.mean():.3f}',f'{t.loc[a,"click_fail"].mean():.3f}']
        rows.append(' & '.join(cells)+r' \\')
    write_table(tab/'web_regimes.tex','@{}lcrrrrrrrr@{}',rows)
    text=(tab/'web_regimes.tex').read_text().replace('\\begin{tabular}{@{}lcrrrrrrrr@{}}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcrrrrrrrr@{}}').replace(r'\end{tabular}',r'\end{tabular*}')
    (tab/'web_regimes.tex').write_text(text)
    # Genuine fit-only baseline and the explicitly labeled hindsight reference.
    d=pd.read_csv(root/'data_compact/corrections/entropy_scale_v1/v3/probability_calibration.csv')
    rows=[r'Model / head & Brier & Fit const. & Test oracle & ECE$_{10}$ \\',r'\midrule']
    for m in MODELS:
        rows.append(r'\multicolumn{5}{@{}l}{'+NAMES[m]+r' ($n=852$)} \\')
        for h,name in [('token_logistic','Token-only logistic'),('geom_token_logistic','Geom.+token logistic'),('geom_gbdt','Geometry GBDT')]:
            r=d[(d.model==m)&(d.split=='test_website')&(d['head']==h)].iloc[0]
            rows.append(f'{name} & {r.brier:.4f} & {r.brier_fit_constant:.4f} & {r.brier_test_oracle:.4f} & {r.ece10:.4f}'+r' \\')
    write_table(tab/'web_calibration.tex','@{}lcccc@{}',rows)
    # Main matched 500-action panel is never mixed into the 852-action table.
    rows=[r'Score & AUROC & C@20 & Coverage & Risk \\',r'\midrule']
    for m,path in [('qwen3vl4b','r9_1'),('internvl35_4b','r9_1/internvl35_4b')]:
        s=json.loads((root/('data_compact/corrections/entropy_scale_v1/r9_'+m)/'eval/r9_summary.json').read_text());nf=round(500*s['test_failure_rate'])
        rows.append(r'\multicolumn{5}{@{}l}{'+NAMES[m]+f': 500 actions, {nf} O2 failures'+r'} \\')
        for h,name in [('geom_gbdt','A: Geometry GBDT'),('full_logistic','A: Full logistic'),('raw_coord_entropy','B: Raw coord. entropy'),
                       ('token_only','B: Token-only logistic'),('geom_token','B: Geom.+token logistic'),
                       ('sg_code_default','C: Dispersion (code)'),('sg_paper_params','C: Dispersion (paper)')]:
            p=s['point_metrics'][h];f=s['frozen_threshold_50pct_calibration'][h]
            rows.append(f'{name} & {p["auroc"]:.3f} & {p["caught_at_20"]} & {f["test_coverage"]:.3f} & {f["test_accepted_risk"]:.3f}'+r' \\')
        rows.append(r'\midrule')
    write_table(tab/'r9_sampling_full.tex','@{}lrrrr@{}',rows[:-1])
    # Exploratory primary and pre-defined CLICK populations: no selected budgets.
    q=pd.read_csv(root/'data_compact/corrections/entropy_scale_v1/v3/queue_metrics.csv');p=pd.read_csv(root/'data_compact/corrections/entropy_scale_v1/v3/paired_queue_contrasts.csv')
    rows=[r'Population / model & Budget ($k$) & O2 caught: O1/O2/T & O1-only: O1/O2/T & Overlap & $\Delta$ caught [95\% CI] \\',r'\midrule']
    for pop in ['all852','CLICK670']:
        for m in MODELS:
            sub=q[(q.model==m)&(q.population==pop)];n=int(sub.n.iloc[0]);nf=int(sub.total_o2_failures.iloc[0])
            rows.append(r'\multicolumn{6}{@{}l}{'+NAMES[m]+f'; {n} actions, {nf} O2 failures'+r'} \\')
            for frac in [.1,.2,.3]:
                ss=sub[sub.fraction==frac].set_index('score');r=p[(p.model==m)&(p.population==pop)&(p.fraction==frac)&(p.score_a=='geom_token_O2')&(p.score_b=='geom_token_O1')].iloc[0]
                hs=['geom_token_O1','geom_token_O2','token_O2']
                caught='/'.join(str(int(ss.loc[h,'caught'])) for h in hs)
                disc='/'.join(str(int(ss.loc[h,'discord_count'])) for h in hs)
                rows.append(f'{"All" if pop=="all852" else "CLICK"} & {int(frac*100)}\\% ({int(r.queue_size)}) & {caught} & {disc} & {int(r.overlap_count)}/{int(r.queue_size)} & {r.delta_caught:+.0f} [{r.delta_caught_lo:.2f},{r.delta_caught_hi:.2f}]'+r' \\')
    write_table(tab/'v3_queues.tex','@{}lccccc@{}',rows)
    # Compact subgroup layout keeps the body readable at final column width.
    rows=[r' & & \multicolumn{2}{c}{Qwen} & \multicolumn{2}{c}{InternVL} \\',r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',r'Subgroup & $n$ & Coverage & Risk & Coverage & Risk \\',r'\midrule']
    p0=root/'data_compact/corrections/entropy_scale_v1/p0'
    for fname,key,groups in [('p0d_by_action_type_frozen.csv','op',['CLICK','TYPE','SELECT']),
                             ('p0d_by_size_tercile_frozen.csv','size_tercile',['small','medium','large'])]:
        d=pd.read_csv(p0/fname)
        for group in groups:
            ss=d[(d[key]==group)&(d['head']=='geom_token_logistic')].set_index('model')
            vals=[v for m in MODELS for v in [f'{ss.loc[m,"frozen_coverage"]:.3f}',f'{ss.loc[m,"frozen_accepted_risk"]:.3f}']]
            label=group + (r'$^*$' if group in ['TYPE','SELECT'] else '')
            rows.append(label+f' & {int(ss.loc[MODELS[0],"n"])} & '+' & '.join(vals)+r' \\')
    write_table(tab/'web_subgroups.tex','@{}lrrrrr@{}',rows)
    # Site table retains the strong token reference, geometry+token, and AUC.
    d=pd.read_csv(p0/'p0d_model_by_website_frozen.csv')
    rows=[r' & & \multicolumn{3}{c}{Qwen3-VL-4B} & \multicolumn{3}{c}{InternVL3.5-4B} \\',
          r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
          r'Website & $n$/tasks & Token C/R & G+T C/R & G+T AUC & Token C/R & G+T C/R & G+T AUC \\',r'\midrule']
    for site in sorted(d.website.unique()):
        cells=[site]
        ss=d[d.website==site];r=ss.iloc[0];cells.append(f'{int(r.n)}/{int(r.tasks)}')
        for m in MODELS:
            for h in ['token_logistic','geom_token_logistic']:
                r=ss[(ss.model==m)&(ss['head']==h)].iloc[0]
                cells.append(f'{float(r.frozen_coverage):.2f}/{float(r.frozen_accepted_risk):.3f}')
            cells.append(f'{float(r.auroc):.3f}')
        rows.append(' & '.join(cells)+r' \\')
    write_table(tab/'web_sites.tex','@{}lccccccc@{}',rows)
