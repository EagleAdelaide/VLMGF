#!/usr/bin/env python3
"""Portable V3 figure/table generation from an extracted artifact.

From artifact root: python src/scripts_rev/v3_figures.py . regenerated
Consumes only shipped data_compact and R9 files, including data_compact/corrections/entropy_scale_v1/v3.
Writes editable SVG and PDF figures, LaTeX tables, and exact data snapshots.
No refitting, GPU inference, or external repository is required.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
                            "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                            "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, VERM, PURPLE, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#8C8C8C", "#56B4E9"
MODELS = [("qwen3vl4b", "Qwen3-VL-4B"), ("internvl35_4b", "InternVL3.5-4B")]
HEADSTYLE = {"geom_gbdt": ("Geometry GBDT (no logits)", ORANGE, ":", "s"),
             "raw_coord_mean_entropy": ("Coordinate-token entropy (raw)", GREY, "-.", "^"),
             "token_logistic": ("Token-only logistic", BLUE, "--", "o"),
             "geom_token_logistic": ("Geometry + token logistic", VERM, "-", "D")}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)

def fig2(repo, paper, snap):
    d = pd.read_csv(repo / "data_compact/corrections/entropy_scale_v1/p0/figure_data/matched_interface_effects.csv")
    d = d[~d.comparison.str.contains("tuple_minus_normalized_json")].copy()
    order = [("qwen3vl4b_refcoco1000", "normalized_json_minus_native_pixel_prompt", "Qwen3-VL-4B · RefCOCO val", "JSON"),
             ("qwen3vl4b_refcoco1000", "normalized_tuple_minus_native_pixel_prompt", "Qwen3-VL-4B · RefCOCO val", "tuple"),
             ("qwen3vl4b_vg1000", "normalized_json_minus_native_pixel_prompt", "Qwen3-VL-4B · Visual Genome", "JSON"),
             ("qwen3vl4b_vg1000", "normalized_tuple_minus_native_pixel_prompt", "Qwen3-VL-4B · Visual Genome", "tuple"),
             ("internvl35_4b_refcoco1000", "normalized_json_minus_native_ref_box", "InternVL3.5-4B · RefCOCO val", "JSON"),
             ("internvl35_4b_refcoco1000", "normalized_tuple_minus_native_ref_box", "InternVL3.5-4B · RefCOCO val", "tuple")]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ys = [5.2, 4.8, 3.7, 3.3, 1.9, 1.5]; rows = []
    for (study, comp, grp, fmt), yv in zip(order, ys):
        r = d[(d.study == study) & (d.comparison == comp)].iloc[0]
        col, mk = (BLUE, "o") if fmt == "JSON" else (ORANGE, "s")
        ax.errorbar(r.mean_delta_iou, yv, xerr=[[r.mean_delta_iou - r.ci95_low], [r.ci95_high - r.mean_delta_iou]], fmt=mk, color=col, ms=4, capsize=2, lw=0.9)
        ax.text(r.ci95_high + 0.012 if r.mean_delta_iou > 0 else r.ci95_low - 0.012, yv, f"{r.mean_delta_iou:+.3f}", ha="left" if r.mean_delta_iou > 0 else "right", va="center", fontsize=6.3, color="#333333")
        rows.append(dict(group=grp, format=fmt, delta=r.mean_delta_iou, lo=r.ci95_low, hi=r.ci95_high, n=int(r.n)))
    ax.set_yticks([5.0, 3.5, 1.7]); ax.set_yticklabels(["Qwen3-VL-4B\nRefCOCO val", "Qwen3-VL-4B\nVisual Genome", "InternVL3.5-4B\nRefCOCO val"]); ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="black", lw=0.7); ax.set_xlim(-0.32, 0.17); ax.set_ylim(0.9, 6.0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("$\\Delta$ mean IoU (normalized package $-$ native package)\nsame 1,000 inputs; paired bootstrap 95% CI")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, lw=0, ms=4, label="normalized JSON"), Line2D([], [], marker="s", color=ORANGE, lw=0, ms=4, label="normalized tuple")],
              loc="upper left", bbox_to_anchor=(0.0, 0.62), frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    ax.text(0.165, 5.95, "helps $\\rightarrow$", ha="right", va="top", fontsize=6.5, color="#555555"); ax.text(-0.03, 5.95, "$\\leftarrow$ hurts", ha="right", va="top", fontsize=6.5, color="#555555")
    pd.DataFrame(rows).to_csv(snap / "fig2_interface_effects.csv", index=False)
    save(fig, paper / "figures/fig2_interface_effects.pdf")

def figA_forest(root, paper, snap):
    co = pd.read_csv(root / "data_compact/joint_coefficients_rev1.csv"); st = pd.read_csv(root / "data_compact/factor_stability_rev1.csv")
    prim = co[(co.analysis == "rev_valid_success_primary") & (co.term.isin(st.term))].merge(st[["term", "stable_rev_gate", "src_boot_ci"]], on="term")
    pretty = {"expression_length_z": "expression length (+1 SD)", "has_spatial": "spatial cue", "has_relation": "relation cue", "has_attribute": "attribute cue", "likely_part": "part reference (large target)",
              "target_edge": "edge target", "border_near": "near-border target", "object_size_medium": "medium target (vs. large)", "object_size_small": "small target (vs. large)",
              "likely_part:object_size_medium": "part × medium", "likely_part:object_size_small": "part × small"}
    order = ["expression_length_z", "has_spatial", "has_relation", "has_attribute", "likely_part", "likely_part:object_size_medium", "likely_part:object_size_small", "object_size_medium", "object_size_small", "target_edge", "border_near"]
    prim = prim.set_index("term").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(3.3, 2.7)); ys = np.arange(len(prim))[::-1]; rows = []
    for yv, (_, r) in zip(ys, prim.iterrows()):
        lo, hi = [float(x) for x in r.src_boot_ci.strip("[]").split(",")]
        stable = bool(r.stable_rev_gate)
        ax.errorbar(r.estimate, yv, xerr=[[r.estimate - lo], [hi - r.estimate]], fmt="o", color=BLUE if stable else GREY, mfc=BLUE if stable else "white", ms=4, capsize=2, lw=0.9)
        rows.append(dict(term=r.term, estimate=r.estimate, lo=lo, hi=hi, stable=stable))
    ax.axvline(0, color="black", lw=0.7); ax.set_yticks(ys); ax.set_yticklabels([pretty[t] for t in prim.term]); ax.set_xlim(-1.9, 0.8)
    ax.set_xlabel("conditional log-odds of IoU$\\geq$0.5\n(model + dataset fixed effects; cluster-bootstrap 95% CI)")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, lw=0, ms=4, label="passes stability gate"), Line2D([], [], marker="o", color=GREY, mfc="white", lw=0, ms=4, label="does not")],
              loc="lower left", frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    pd.DataFrame(rows).to_csv(snap / "figA_factor_forest.csv", index=False)
    save(fig, paper / "figures/figA_factor_forest.pdf")

def figA_rankcorr(repo, paper, snap):
    d = pd.read_csv(repo / "data_compact/corrections/entropy_scale_v1/p0/figure_data/integrated_nine_vlm_results.csv")
    piv = d.pivot(index="slug", columns="family_key", values="mean_iou"); assert piv.shape == (9, 5), piv.shape
    corr = piv.rank(ascending=False).corr(method="pearson")  # Spearman as Pearson on ranks (as in the original script)
    fam = [("refcoco", "RefCOCO"), ("v7w", "Visual7W"), ("vg", "VG"), ("flickr", "Flickr30k"), ("gqa", "GQA")]
    keys = [k for k, _ in fam]; corr = corr.loc[keys, keys]
    ci = pd.read_csv(repo / "data_compact/corrections/entropy_scale_v1/p0/figure_data/spearman_summary.csv")
    m2r = {"refcoco": "refcoco", "v7w": "visual7w", "vg": "visual_genome", "flickr": "flickr30k_entities", "gqa": "gqa"}
    look = {frozenset([r.family_a, r.family_b]): (r.spearman_ci_low, r.spearman_ci_high) for r in ci.itertuples()}
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); im = ax.imshow(corr.to_numpy(), vmin=0, vmax=1, cmap="Greys", aspect="equal")
    ax.set_xticks(range(5)); ax.set_xticklabels([n for _, n in fam], rotation=30, ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels([n for _, n in fam])
    rows = []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            v = float(corr.loc[a, b])
            if i == j:
                txt = "1.00"
            else:
                lo, hi = look[frozenset([m2r[a], m2r[b]])]; txt = f"{v:.2f}\n[{lo:.2f},{hi:.2f}]"; rows.append(dict(a=a, b=b, rho=v, lo=lo, hi=hi))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.0, color="white" if v > 0.6 else "black")
    ax.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False, ec="black", lw=1.2)); ax.add_patch(Rectangle((1.5, 1.5), 3, 3, fill=False, ec="black", lw=1.2))
    ax.tick_params(length=0); fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("Spearman $\\rho$ (9 models)", fontsize=7)
    pd.DataFrame(rows).to_csv(snap / "figA_rankcorr.csv", index=False)
    save(fig, paper / "figures/figA_rankcorr.pdf")


def main():
    # Explicit sibling imports also work under python -I from any extraction path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import v3_panels as V
    import v3_tables as T
    root, out = [Path(a).resolve() for a in sys.argv[1:3]]
    snap = out / 'figure_data'; snap.mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    print('EXCLUDED: Figure 1 screenshot illustration; numerical panels are provided as CSVs.')
    V.selective(root, out); V.sampling(root, out, snap)
    V.budget(root, out, snap)
    V.interfaces(root, out, snap); figA_forest(root, out, snap); figA_rankcorr(root, out, snap)
    T.tables(root, out)
    # Exact plotting inputs, including pointwise bands and fixed operating points.
    import shutil
    for f in (root / 'data_compact/corrections/entropy_scale_v1/v3').glob('*.csv'):
        if f.name not in ['exploratory_scores.csv','queue_ids.csv']:
            shutil.copyfile(f, snap / f.name)
    for f in ['p0b_outcome_contingency_852.csv','p0b_same_accepted_set_two_outcomes.csv']:
        shutil.copyfile(root / 'data_compact/corrections/entropy_scale_v1/p0' / f, snap / f)
    print('Portable V3 figures and tables written to', out)


if __name__ == '__main__':
    main()
