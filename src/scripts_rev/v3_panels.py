"""V3 figure panels. Imported by the portable v3_figures.py command."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

BLUE, ORANGE, GREEN, VERM, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#777777'
MODELS = [('qwen3vl4b', 'Qwen3-VL-4B'), ('internvl35_4b', 'InternVL3.5-4B')]
STYLES = [('geom_gbdt','Geometry GBDT', ORANGE, ':', 's'),
          ('raw_coord_mean_entropy','Raw coordinate entropy', GREY, '-.', '^'),
          ('token_logistic','Token-only logistic', BLUE, '--', 'o'),
          ('geom_token_logistic','Geometry + token logistic', VERM, '-', 'D')]


def save(fig, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(p.with_suffix('.svg'), metadata={'Date': None})
    plt.close(fig)


def outcome_schematic_superseded(root, out):
    p0 = root / 'data_compact/corrections/entropy_scale_v1/p0'
    ct = pd.read_csv(p0 / 'p0b_outcome_contingency_852.csv').set_index('model')
    acc = pd.read_csv(p0 / 'p0b_same_accepted_set_two_outcomes.csv').set_index('model')
    fig = plt.figure(figsize=(7, 2.75))
    a = fig.add_axes([.015,.43,.235,.41]); a.set(xlim=(0,100),ylim=(0,100)); a.axis('off')
    a.add_patch(Rectangle((2,10),95,80,facecolor='#f5f5f5',edgecolor='#bbb',lw=.7))
    a.add_patch(Rectangle((48,43),44,19,facecolor='white',edgecolor=GREEN,lw=1.7))
    a.text(70,52,'Target',ha='center',va='center',fontsize=8)
    a.add_patch(Rectangle((59,47),19,10,facecolor='none',edgecolor=ORANGE,ls='--',lw=1.2))
    a.add_patch(Rectangle((37,28),60,48,facecolor='none',edgecolor=BLUE,ls='--',lw=1.2))
    a.add_patch(Rectangle((8,43),23,18,facecolor='none',edgecolor=VERM,ls='--',lw=1.2))
    for x,y,c in [(69,52,ORANGE),(67,52,BLUE),(19.5,52,VERM)]: a.plot(x,y,'o',color=c,ms=3.5)
    fig.text(.015,.93,'(a) Schematic, no data',weight='semibold',fontsize=8.5)
    a.legend(handles=[Line2D([],[],color=ORANGE,ls='--',label='Small: O1 fail, O2 pass'),
                      Line2D([],[],color=BLUE,ls='--',label='Broad: O1 fail, O2 pass'),
                      Line2D([],[],color=VERM,ls='--',label='Off target: both fail')],
             loc='upper left',bbox_to_anchor=(0,-.40),frameon=False,fontsize=7.5,
             handlelength=1.3,borderpad=0,labelspacing=.3)
    b = fig.add_axes([.31,.43,.31,.41])
    for i,(m,name) in enumerate(MODELS):
        y = 1.1-i
        b.text(0,y+.30,name,fontsize=7.8)
        left=0
        for col,color in [('both_ok',GREEN),('iou_fail_point_hit',ORANGE),('both_fail',VERM)]:
            n=int(ct.loc[m,col]); width=n/852*100
            b.barh(y,width,left=left,height=.48,color=color,edgecolor='white',lw=.6)
            b.text(left+width/2,y,f'{n/852*100:.1f}%\n({n})',ha='center',va='center',
                   fontsize=7.5,color='black' if color==ORANGE else 'white')
            left+=width
    b.set(yticks=[],xticks=[0,50,100],xlim=(0,100),ylim=(-.25,1.65),xlabel='All 852 actions (%)')
    b.spines['left'].set_visible(False)
    fig.text(.31,.93,'(b) Measured contingency',weight='semibold',fontsize=8.5)
    b.legend(handles=[Line2D([],[],color=GREEN,lw=5,label='Both pass'),
                      Line2D([],[],color=ORANGE,lw=5,label='O1 fail, O2 pass'),
                      Line2D([],[],color=VERM,lw=5,label='Both fail')],loc='upper left',
             bbox_to_anchor=(0,-.40),frameon=False,fontsize=7.5,handlelength=1,borderpad=0,labelspacing=.3)
    c=fig.add_axes([.72,.43,.26,.41])
    for i,(m,name) in enumerate(MODELS):
        r=acc.loc[m];y=1.1-i
        c.text(0,y+.36,name,fontsize=7.8)
        for key,color,marker,dy in [('o2',GREEN,'o',.07),('o1',BLUE,'s',-.07)]:
            v=float(r.o2_point_fail_rate if key=='o2' else r.o1_iou_fail_rate)*100
            lo=float(r[key+'_lo'])*100; hi=float(r[key+'_hi'])*100
            c.errorbar(v,y+dy,xerr=[[v-lo],[hi-v]],fmt=marker,color=color,ms=4.5,capsize=2,lw=1)
            c.text(v,y+(.17 if key=='o2' else -.29),f'{v:.1f}%',ha='center',fontsize=7.5,color=color)
    c.set(yticks=[],xticks=[0,30,60],xlim=(0,72),ylim=(-.30,1.7),xlabel='Accepted-set failure (%)')
    c.spines['left'].set_visible(False)
    fig.text(.72,.93,'(c) Same accepted IDs',weight='semibold',fontsize=8.5)
    c.legend(handles=[Line2D([],[],color=GREEN,marker='o',lw=0,label='O2: center outside'),
                      Line2D([],[],color=BLUE,marker='s',lw=0,label='O1: IoU < 0.5')],loc='upper left',
             bbox_to_anchor=(0,-.40),frameon=False,fontsize=7.5,borderpad=0,labelspacing=.3)
    fig.text(.5,.025,'O1 success implies O2 success for axis-aligned boxes (geometrically expected).',
             ha='center',fontsize=7.5)
    save(fig,out/'figures/fig1_outcome_mismatch.pdf')


def selective(root,out):
    d=pd.read_csv(root/'data_compact/corrections/entropy_scale_v1/v3/risk_coverage_pointwise.csv')
    met=pd.read_csv(root/'data_compact/corrections/entropy_scale_v1/v3/singlepass_metrics.csv')
    fig=plt.figure(figsize=(7,2.45))
    for j,(m,name) in enumerate(MODELS):
        ax=fig.add_axes([.09+j*.49,.40,.39,.49])
        for h,label,color,ls,mk in STYLES:
            b=d[(d.model==m)&(d['head']==h)]
            ax.fill_between(b.coverage,b.lo,b.hi,color=color,alpha=.12,lw=0)
            ax.plot(b.coverage,b.risk,color=color,ls=ls,lw=1.2,label=label)
            r=met[(met.model==m)&(met['head']==h)&(met.population=='all852')].iloc[0]
            ax.plot(r.coverage,r.risk,marker=mk,color=color,
                    mfc='none' if h=='raw_coord_mean_entropy' else color,
                    mec=color if h=='raw_coord_mean_entropy' else 'black',
                    mew=.9 if h=='raw_coord_mean_entropy' else .6,
                    ms=7 if h=='raw_coord_mean_entropy' else 5,ls='',
                    zorder=8 if h=='raw_coord_mean_entropy' else 5)
        base=188/852 if j==0 else 541/852
        ax.axhline(base,color='black',ls=(0,(2,2)),lw=.8)
        ax.set(xlim=(.05,1),ylim=(0,.30 if j==0 else .73),xlabel='Accepted fraction',
               ylabel='O2 failure rate' if j==0 else '')
        ax.set_title(f'({"ab"[j]}) {name}',loc='left',fontsize=8.5)
        ax.grid(alpha=.25,lw=.4)
    handles=[Line2D([],[],color=c,ls=ls,marker=mk,ms=5 if h=='raw_coord_mean_entropy' else 4,
             markerfacecolor='none' if h=='raw_coord_mean_entropy' else c,lw=1.2,label=lab) for h,lab,c,ls,mk in STYLES]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.53,.125),ncol=2,frameon=False,
               fontsize=8,handlelength=2.4,columnspacing=2,labelspacing=.5)
    fig.text(.53,.02,'Markers: frozen median thresholds. Black dashed: accept-all risk.',ha='center',fontsize=7.5)
    save(fig,out/'figures/fig3_web_selective_review.pdf')


def sampling(root,out,snap):
    rows=[]
    for m,path in [('qwen3vl4b','r9_1'),('internvl35_4b','r9_1/internvl35_4b')]:
        s=json.loads((root/('data_compact/corrections/entropy_scale_v1/r9_'+m)/'eval/r9_summary.json').read_text())
        # Paired intervals are supplied by the original R9 evaluator.
        for params in ['code','paper']:
            item=s['deltas']['sg_' + ('code_default' if params=='code' else 'paper_params') + '-minus-token_only']
            rows.append(dict(model=m,params=params,delta=item['point'],lo=item['lo'],hi=item['hi']))
    d=pd.DataFrame(rows);d.to_csv(snap/'fig4_paired_sampling_differences.csv',index=False)
    fig=plt.figure(figsize=(3.42,2.8))
    ax=fig.add_axes([.34,.39,.62,.47])
    for j,(m,name) in enumerate(MODELS):
        for params,offset,mk,c in [('code',.13,'o',BLUE),('paper',-.13,'s',ORANGE)]:
            r=d[(d.model==m)&(d.params==params)].iloc[0];y=1-j+offset
            ax.errorbar(r.delta,y,xerr=[[r.delta-r.lo],[r.hi-r.delta]],fmt=mk,color=c,ms=4.7,capsize=2,lw=1.1)
    ax.axvline(0,color='black',lw=.8)
    ax.set(xlim=(-.125,.115),ylim=(-.4,1.4),yticks=[1,0],
           yticklabels=['Qwen3-\nVL-4B','InternVL\n3.5-4B'],xticks=[-.1,0,.1],
           xlabel='AUROC difference vs. token-only')
    ax.tick_params(axis='y',length=0); ax.grid(axis='x',alpha=.25)
    fig.text(.02,.94,'Same 500 actions; O2 failures: 115 / 328',fontsize=8)
    fig.legend(handles=[Line2D([],[],marker='o',lw=1,color=BLUE,label='Released-code defaults'),
                        Line2D([],[],marker='s',lw=1,color=ORANGE,label='Published parameters')],
               loc='center',bbox_to_anchor=(.53,.19),ncol=2,frameon=False,fontsize=7.5,handlelength=1.2,columnspacing=1)
    fig.text(.02,.09,'Mean seconds/input    Greedy    +10 samples    Total/base',fontsize=7.5)
    fig.text(.02,.045,'Qwen / InternVL       1.49/1.42      8.26/3.15        6.53/3.21',fontsize=7.5)
    save(fig,out/'figures/fig4_cost_tradeoff.pdf')


def budget(root,out,snap):
    d=pd.read_csv(root/'data_compact/corrections/entropy_scale_v1/p0/figure_data/figA_label_budget.csv')
    fig,ax=plt.subplots(figsize=(3.42,2.7),layout='constrained')
    for h,lab,c,ls,mk in [('full_gbdt','Full GBDT',BLUE,'-','o'),('geom_gbdt','Geometry GBDT',ORANGE,'-','s'),
                          ('full_logistic','Full logistic',BLUE,'--','o'),('geom_logistic','Geometry logistic',ORANGE,'--','s')]:
        a=d[d['head']==h].sort_values('budget')
        ax.plot(a.budget,a.macro_mean,color=c,ls=ls,marker=mk,ms=3,label=lab)
        ax.fill_between(a.budget,a.macro_mean-a.seed_sd,a.macro_mean+a.seed_sd,color=c,alpha=.12,lw=0)
    ax.set(xscale='log',xlabel='Fitting labels (log scale)',ylabel='Macro failure AUROC',ylim=(.55,.78))
    ax.set_xticks([50,100,200,500,1000,2000,4000],['50','100','200','500','1k','2k','4k']);ax.minorticks_off()
    ax.legend(frameon=False,fontsize=7.5);ax.grid(alpha=.2)
    d.to_csv(snap/'figA_label_budget.csv',index=False)
    save(fig,out/'figures/label_budget_natural.pdf')


def interfaces(root,out,snap):
    d=pd.read_csv(root/'data_compact/corrections/entropy_scale_v1/p0/figure_data/matched_interface_effects.csv')
    spec=[('qwen3vl4b_refcoco1000','normalized_json_minus_native_pixel_prompt'),
          ('qwen3vl4b_refcoco1000','normalized_tuple_minus_native_pixel_prompt'),
          ('qwen3vl4b_vg1000','normalized_json_minus_native_pixel_prompt'),
          ('qwen3vl4b_vg1000','normalized_tuple_minus_native_pixel_prompt'),
          ('internvl35_4b_refcoco1000','normalized_json_minus_native_ref_box'),
          ('internvl35_4b_refcoco1000','normalized_tuple_minus_native_ref_box')]
    fig=plt.figure(figsize=(3.42,2.25));ax=fig.add_axes([.34,.32,.63,.63]);rows=[]
    for i,(study,comparison) in enumerate(spec):
        r=d[(d.study==study)&(d.comparison==comparison)].iloc[0]
        c,mk=(BLUE,'o') if i%2==0 else (ORANGE,'s');y=[5.2,4.8,3.2,2.8,1.2,.8][i]
        ax.errorbar(r.mean_delta_iou,y,xerr=[[r.mean_delta_iou-r.ci95_low],[r.ci95_high-r.mean_delta_iou]],
                    fmt=mk,color=c,ms=4,capsize=2,lw=1)
        rows.append(r.to_dict())
    ax.axvline(0,color='black',lw=.8)
    ax.set(xlim=(-.30,.15),ylim=(.2,5.8),yticks=[5,3,1],xticks=[-.3,-.15,0,.15],
           yticklabels=['Qwen3-VL-4B\nRefCOCO','Qwen3-VL-4B\nVisual Genome','InternVL3.5-4B\nRefCOCO'],
           xlabel='$\\Delta$ mean IoU (normalized $-$ native)')
    ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.2)
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',lw=0,label='JSON'),
                        Line2D([],[],color=ORANGE,marker='s',lw=0,label='Tuple')],
               loc='center',bbox_to_anchor=(.63,.06),ncol=2,frameon=False,fontsize=7.5)
    pd.DataFrame(rows).to_csv(snap/'fig2_interface_effects.csv',index=False)
    save(fig,out/'figures/fig2_interface_effects.pdf')


def outcome(root,out):
    """Keep the verified archived Figure 1 unchanged; never synthesize the input."""
    import shutil
    source=root/'figure_sources/fig1_refined'
    (out/'figures').mkdir(exist_ok=True,parents=True)
    for ext in ['pdf','svg']:
        shutil.copyfile(source/('fig1_refined_candidate.'+ext),out/'figures'/('fig1_refined_candidate.'+ext))
