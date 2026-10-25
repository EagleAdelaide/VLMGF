# V3 exploratory amendment — 21 October 2026 (original dated protocol)

Written before computing the new queue contrasts. Earlier scores, labels, AUROCs,
and the independent review have already been inspected. This is an exploratory
amendment, not a preregistration or an untouched confirmatory study.

Starting point: `8e86227` on `www2027/p0p2-revision` (parent `506ab6e`).
The original candidate and all frozen model outputs, labels, scores, thresholds,
sampling results, and failed research-track gates remain preserved.

Question: does fitting to O1 instead of O2 change limited review queues, or does
the outcome definition chiefly change accounting for the same predictions?

Freeze before computation:

- Primary population: the identical 852 eligible `test_website` actions for each
  of Qwen3-VL-4B and InternVL3.5-4B. Separately report the predefined 670 CLICK
  actions. `test_domain` (504 actions) is secondary and cannot select a result.
- Review fractions: exactly 0.10, 0.20, 0.30; queue size is `ceil(fraction*N)`.
- Two existing logistic feature specifications only: geometry + token, and
  token only (the strong reference). StandardScaler and LogisticRegression
  (C=1, L2, lbfgs, max_iter=2000, random_state=20261009), identical to the
  frozen CPU reproduction. Fit on the same 510 actions, changing only the
  success label (1-iou_fail versus 1-click_fail). Fit-only token median
  imputation and the original missingness indicators. No test labels enter
  preprocessing, fitting, hyperparameters, or thresholds. Refit O2 solely to
  check agreement with frozen scores; require maximum difference <1e-6.
- If paired O1 per-row scores are not shipped, create isolated CPU refits;
  never replace legacy files. No alternative learners, budget search, subgroup
  search, new VLM inference, or simulated human audit.
- One deterministic label-independent tie key for all V3 ranking outputs:
  SHA-256 of `VLMGF-V3-20261021|sample_id`, interpreted as a hexadecimal
  string. IDs are unique within each split. In bootstrap duplicates retain
  the same key; their outcomes/scores are identical. Legacy R9 uses its
  original SEED+7 tie sequence and remains explicitly labeled as legacy.
- For each score report O2 failures caught, denominator, queue size, and
  O1-failure/O2-success count and fraction. Compare O2-fit minus O1-fit
  within each feature specification; compare O2 geometry+token minus the
  O2 token-only reference. Report overlap count, overlap/queue size, Jaccard,
  and paired caught-count and recall differences.
- 2,000 paired task-cluster bootstrap draws, seed 20261021. For each
  population resample its task universe once per draw, synchronously across
  both models and every score, rerank each resampled action multiset and
  apply the same fraction. Percentile 95% intervals are pointwise,
  conditional on these fixed fitted heads, with no multiplicity correction
  and no retraining uncertainty. Variable bootstrap population sizes make
  caught-count intervals conditional on the cluster-resampling scheme;
  paired recall differences are also reported.
- Fixed-score O1/O2 relabeling must produce identical queue IDs. The new
  analysis concerns changes in *fitted labels*, not relabeling a fixed score.
  O2 success in a queue is not evidence of wasted human effort: neither
  human review nor browser action execution was performed.
- Secondary descriptive robustness: on original frozen O2 scores, freeze
  calibration-score quantiles 0.25/0.50/0.75 using the 200 scored inputs,
  and report realized coverage/risk by website. No threshold selection or
  future-site risk guarantee. Median thresholds remain unchanged.

Interpretation: retain unresolved or adverse queue contrasts. Do not promote
one favorable budget or subgroup. Separate outcome-sensitive accounting from
decision value; no equivalence claim or acceptance prediction is authorized.
