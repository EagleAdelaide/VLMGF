# FINAL_AUDIT_PROTOCOL (frozen 2026-10-10, before any of the analyses below was run)

> English translation of the frozen protocol, made 2026-10-17 for the anonymous package.
> Sections A-F (the analysis specifications) are translated in full; internal work-item
> labels are omitted, and the original's section G (a list of manuscript edits, not an
> analysis specification) is not included. No specification was changed.

Specifications below were frozen first and executed afterwards; all new analyses are CPU-only with 0 new GPU calls; test_website/test_domain are evaluation-only and are never used for any selection.

## A. Random-baseline Monte Carlo
- Frozen after checking the implementation: review budget k = ceil(frac·n) (consistent with the existing `np.ceil` implementation); tie-break = the seeded uniform order applied before the budget ranking (the random order itself breaks ties).
- For each model×split×outcome: 10,000 uniform random orderings **without replacement** (numpy default_rng(20261010), draw i uses a permutation); report the mean and the [2.5, 97.5] percentiles of AUROC, AURC, Risk@50 and C@20.
- Theoretical expectations listed separately: E[C@f] = k_f·n_fail/n (k_f = ceil(f·n)); E[Risk@c] = prevalence; E[AUROC] = 0.5.
- The random row of the main table = the MC mean; the interval goes into the text/footnote; the earlier single-draw value is labelled in the ledger as a realized draw.

## B. Stronger Web baselines
- New heads (all with the R5 frozen GBDT grid depth{2,3}×iters{100,200}, lr 0.1, parameters selected by **3-fold CV within task groups**; seed 20261009): token_gbdt, geom_token_gbdt (geom_gbdt/full_gbdt already exist). Feature sets unchanged (R7/E3 definitions; no GT, labels or website id in the features).
- Fitting on fit-510 only; calib-200 only for freezing thresholds; both test splits evaluation-only.
- Paired Δ (synchronized task-group bootstrap, 2,000 draws, seed 20261009, macro-average over the two models):
  geom_token−geom_gbdt (vs. the strongest geometry head), geom_token−token_only, geom_token_gbdt−geom_gbdt, geom_token_gbdt−token_gbdt, token_gbdt−geom_gbdt.
- Interpretation rule (frozen): if a Δ's 95% CI contains 0, the text must not call that comparison "significant/best" and uses tied/CI-crossing wording instead.

## C. test_domain reporting
- Directly reuse the existing per-example results plus this round's new heads; WEB_SPLIT_COMPARISON.csv puts the two splits side by side (n, websites, tasks, prevalence, AUROC/AURC/C@20/Risk@50, MC random, CI).
- Calib-frozen thresholds applied to domain: the rule is frozen on calib (same thresholds as for website) and labelled as a second evaluation of the same frozen rule (not a re-tuning).

## D. Site sensitivity
- Per-website table (website names kept: they are the official public split names and carry no anonymity risk; action/task counts, prevalence, AUROC/AURC/C@20 where computable, N/A for single-class outcomes).
- Leave-one-website-out, evaluation-only: fixed fitted scores and frozen thresholds; drop one site at a time and recompute the aggregates (AUROC, C@20, realized risk/coverage for geom_token and token_only); report min/median/max; not used for any selection.
- Website-cluster bootstrap (10 clusters) as sensitivity only; the text states its coarseness.

## E. Eligibility audit and subset sensitivity
- Per-split atomic and mutually exclusive exclusion counts (exclusions.csv attributed by split); distributions of operation type, website and task.
- GT-in-crop check: share with rect_bottom = y+h ≤ 4096; partial-in-crop count (center inside, bottom edge outside).
- Pre-defined sensitivity subsets (exploratory, fixed here): S1 = CLICK-only; S2 = GT entirely inside the crop; S3 = label length ≥ 4 characters. For each subset recompute prevalence and the geom_token/token AUROC (filtering on the evaluation side).
- QA montage: seed 20261010, stratified sampling (per model×{click-ok, click-fail}×{iou-fail-click-ok}×edge), blinded output (no scores/methods shown), ≥24 box-overlay images, for manual review by the authors; no claim is made that a blind human review has passed.

## F. Number-consistency checker
- Extend audit_number_check.py to cover all new tables and the paper's revised numbers; non-zero exit on any failure.
