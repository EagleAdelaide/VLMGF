# Beyond IoU: reproducibility materials

Outcome-aware reliability evaluation and failure triage for Web grounding. The study uses **static screenshot target localization**, oracle-provided target descriptions and already-generated box predictions. It does not execute browser actions or correct boxes.

Reviewer access: https://anonymous.4open.science/r/VLMGF-27C5/

## Clean CPU setup

Use CPython **3.11**, a CPU machine and several GB of free disk/RAM. From the downloaded repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -B -I repro/release_smoke.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -B -I repro/reproduce.py --out-dir ../vlmgf-replay
```

The lockfile pins the clean tested environment, including `pyarrow` for Parquet. `requirements.txt` retains the eight frozen direct dependency versions. Installation needs a package index; CPU replay itself needs no network, model download or GPU. The full replay includes 2,000-draw grouped bootstraps, fitted heads, conditional factor refits and saved-output dispersion scoring, so allow more time than the seconds-long smoke test. The clean local full replay passed in 430 seconds with four BLAS/OpenMP threads; see `REPRODUCTION_REPORT.json`. Hardware and filesystem affect runtime; peak memory was not instrumented. Use a new output path **outside** this repository; existing results are never publication inputs.

## Reproduction levels

- **L1:** recompute metrics and uncertainty intervals from the shared per-example outcomes/scores and group IDs. `repro/reproduce_core.py --level L1` performs the core grouped bootstrap/ranking checks. The full wrapper additionally reconstructs corrected raw entropy, recalculates frozen operating points, accepted IDs, matched 500-action dispersion and exploratory review queues.
- **L2:** refit the eligible analysis heads and conditional factor model from frozen features/folds. `repro/reproduce_core.py --level L2` runs the core refits; the full wrapper runs L1/L2 together and the declared sensitivity/refit comparisons. These are reproductions of frozen analyses, not new model inference.
- **L3:** regenerating original VLM outputs is **not supported** here; obtain licensed images and checkpoints from the official providers and supply GPUs. See `DATA_PROVENANCE_AND_LIMITS.md`.

The full wrapper writes `logs/` and `REPLAY_REPORT.json` in the selected output directory, plus regenerated numerical CSVs and six statistical figure PDFs/six generated table sources. Figure 1's third-party screenshot illustration is **not redistributed**; its outcome/accepted-set numerical inputs are included. The omission is explicit and is never counted as a successful screenshot regeneration.

To regenerate statistical plots/tables only:

```bash
.venv/bin/python -B -I src/scripts_rev/v3_figures.py . ../vlmgf-plots
```

## Frozen rules and expected agreement

Raw coordinate entropy is the **unnormalized** entropy H, with fit-only median imputation and a single calibration-median threshold applied unchanged to test inputs, including ties. Never normalize calibration and test batches independently. Corrected authoritative files are in `data_compact/corrections/entropy_scale_v1/`; historical raw-score columns retained for regression comparisons are not current operating-point evidence.

The machine-readable `CLAIM_LEDGER.csv` and `FIGURE_TABLE_CLAIM_MAP.csv` map results to populations, inputs and uncertainty. `repro/release_smoke.py` recomputes the principal Web outcomes, accepted sets and corrected operating points against these frozen derived ledgers. The original validators declare each numerical tolerance: core reported point estimates generally use 5e-4 or their printed precision; paired bootstrap CI tolerances are explicit in `repro/reproduce_core.py`; full derived CSV comparisons use `rtol=1e-13, atol=1e-14`; unchanged statistical PDF/table outputs are compared by bytes under the pinned environment. Do not reduce bootstrap draws or skip GBDT and call that full validation.

Every comparison names its population. The two-model Web samples include 510 fitting and 200 calibration actions, 852 held-out-website actions, a separately sampled 504-action held-out-domain check and a matched 500-action sampling panel. Model-specific sampling parameters and generation/implementation-specific latency remain in the protocols and saved sample records. Opposite dispersion directions do not imply a separately tested interaction. Queue changes do not establish additional caught-failure yield; intervals containing zero do not establish equivalence.

## Files and exclusions

`src/`, `repro/` and `r9_1/` contain the CPU code, parser, saved coordinate samples and vendored scorer. `data_compact/`, `protocol/`, `prompt_repair/` and `human_audit/` contain frozen numeric inputs and scoped prior audit evidence; the human audit concerns non-Web items. `figures_revised/`, `tables_revised/` and `figure_data_revised/` contain the corrected statistical exhibits and plotting inputs. Exact code/data identifiers remain in executable inputs and narrowly scoped protocol descriptions.

`RELEASE_INVENTORY.json` and `checksums.sha256` inventory the public tree; they never include `.git`, credentials, temporary outputs or a manifest's own hash. `PUBLIC_RELEASE_EXCLUSIONS.json` explains omitted private packaging/history, infrastructure logs, unsupported source-data ingestion scripts and restricted imagery. The public Web manifest removes four unused text/value columns; IDs, numeric fields, evaluation membership and order remain intact.

See `LICENSE.txt`, the preserved SafeGround license and `DATA_PROVENANCE_AND_LIMITS.md` for scope, third-party attribution and acquisition instructions. No separate OpenReview ZIP upload is assumed.

The three remaining numeric table-cell sources are preserved in `tables_revised/` and traced to their frozen CSV/JSON inputs in the claim map. The two protocol/population tables are manuscript reporting specifications, not new statistical analyses. `src/scripts_rev/run_final_audit_web.py` is retained to document the original producer feature definition; the supported portable execution entry points are the CPU commands above.
