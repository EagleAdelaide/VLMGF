# Prompt-level repair tests (negative result)

Supports the Discussion sentence on "the unsuccessful prompt-level repair tests".
These tests asked whether per-sample prompt choices can produce a better box for an
individual prediction. They are a negative record, not a proposed method. File and
script names carry the internal label `far_ground`.

## Scope

| | M1: per-sample interface selection | M2: repair prompts |
|---|---|---|
| Models | Qwen3-VL-4B, InternVL3.5-4B | same |
| Population | the matched RefCOCO val 1,000 expressions of the paper's crossed-interface study (Sec. 5.1) | 200 RefCOCO val samples from the M1 population, one per unique image, proxy-stratified (quotas in `m2_repair_prompts/operator_protocol.md`) |
| Candidates / operators | native(-like), normalized JSON, normalized tuple | relation-aware and part-aware prompts (`m2_repair_prompts/prompts.md`) |
| Fixed baseline | best global fixed interface: Qwen tuple (mIoU 0.8467, IoU@0.5 0.918); InternVL native (0.8517, 0.921) | each model's best fixed M1 interface on the same 200 samples: Qwen tuple (mIoU 0.8374); InternVL native (0.8338) |
| Selection inputs | test-time features only (expression cues, candidate geometry/validity, cross-candidate agreement); five-fold `GroupKFold` by source image | target subsets fixed by rule-based eligibility flags (relation/spatial: 131; part reference: 52) |
| Extra model calls | none (reuses existing predictions) | one per sample per operator: 800 generations |
| Execution status | **executed; gate failed for both models** | **executed; all four model-by-operator gates failed** |

M3 (method implementation) and M4 (main evaluation) were **not run**: they were
conditional on M1/M2 passing. They are not failed experiments.

## Gates (as implemented in `src/`)

- M1, per model: oracle minus best fixed interface ≥ +0.025 mean IoU or
  ≥ +0.035 IoU@0.5 (`headroom_pass`); grouped router recovers ≥ 25% of the
  oracle mean-IoU gap (`router_pass`); headroom also holds on rows where all candidates
  parse (`parse_independent_headroom`). All three are required.
- M2, per model-by-operator: target-subset gain ≥ +0.03 mean IoU or ≥ +0.05
  IoU@0.5 (`target_pass`); non-target mean-IoU change ≥ -0.01 (`non_target_pass`);
  positive target gain on rows where baseline and repair both parse, covering
  ≥ 75% of the target subset (`parse_independent`). All three are required. Paired
  bootstrap intervals are reported; an interval excluding zero was not a gate
  criterion.

Gates, prompts, and sample membership were recorded in the project protocol before the
corresponding outputs were inspected (author-reported; this artifact cannot
independently timestamp that order).

## M1 results (`m1_interface_selection/gate_decision.csv`, `bootstrap.csv`)

| Model | Oracle − best fixed: ΔmIoU [95% CI] | ΔIoU@0.5 | Full router ΔmIoU [95% CI] | Recovery | Factor-only router ΔmIoU [95% CI] | Valid-only oracle gap | Gate |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B | +0.0176 [+0.0126, +0.0235] | +0.016 | −0.0017 [−0.0071, +0.0037] | −9.5% | −0.0005 [−0.0036, +0.0029] | +0.0177 (n=995) | FAIL |
| InternVL3.5-4B | +0.0191 [+0.0136, +0.0252] | +0.024 | −0.0013 [−0.0048, +0.0015] | −6.9% | −0.0004 [−0.0011, +0.0000] | +0.0191 (n=999) | FAIL |

The oracle uses ground-truth IoU and is an unattainable upper bound. Per-interface
win shares by factor level are in `factor_conditioned_best.csv`; per-sample choices of
every strategy are in `strategy_predictions.csv` (`group_id` = source image).

## M2 results (`m2_repair_prompts/operator_gate.csv`, `operator_bootstrap.csv`)

| Model | Operator | Target n | Target ΔmIoU [95% CI] | Target ΔIoU@0.5 | Non-target n | Non-target ΔmIoU [95% CI] | target / non-target / parse-indep. | Gate |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B | relation | 131 | −0.0130 [−0.0386, +0.0097] | −0.0153 | 69 | −0.0086 [−0.0231, +0.0005] | fail / pass / fail | FAIL |
| Qwen3-VL-4B | part | 52 | −0.0040 [−0.0201, +0.0063] | +0.0000 | 148 | −0.0135 [−0.0277, −0.0011] | fail / fail / fail | FAIL |
| InternVL3.5-4B | relation | 131 | +0.0038 [−0.0098, +0.0185] | +0.0229 | 69 | −0.0124 [−0.0355, +0.0048] | fail / fail / pass | FAIL |
| InternVL3.5-4B | part | 52 | +0.0237 [−0.0089, +0.0702] | +0.0577 | 148 | −0.0152 [−0.0367, +0.0025] | pass / fail / pass | FAIL |

All 800 repair outputs and all baseline rows on these 200 samples parsed. InternVL
part repair met the target criterion through IoU@0.5 but changed non-target mean IoU by
−0.0152, beyond the −0.01 limit; Qwen relation repair stayed within the non-target
limit but did not improve its target subset. Whole-sample (n=200) changes and
improved/worsened/unchanged counts are in `operator_bootstrap.csv`; per-sample paired
values are in `paired_operator_samples.csv`.

## Uncertainty

M1 intervals: 5,000 bootstrap resamples of expressions (seed 20260715); the router's
cross-validation folds are image-grouped, but the interval resampling is not. M2
intervals: 5,000 paired bootstrap resamples of the 200 samples (seed 20260715), each a
distinct image.

## Reading

Under the evaluated protocols, neither per-sample selection among the three prompt
interfaces nor the two repair prompts met the joint criteria. This does not mean that
every metric declined (favorable cells are listed above) or that other repair
strategies would fail; none was evaluated.

## Files

- `m1_interface_selection/`: gate decision, strategy summary, bootstrap intervals,
  factor-conditioned win shares, candidate complementarity, error overlap, per-sample
  strategy choices, run config.
- `m2_repair_prompts/`: frozen protocol, verbatim prompts, sample manifest with proxy
  strata and eligibility flags, gate table, bootstrap table, absolute metrics,
  per-sample paired deltas.
- `src/`: the analysis and sample-construction scripts. They read raw prediction files
  that are not shipped; `repro/check_repair_and_audit.py` re-derives every gate flag
  above from the shipped CSVs.
