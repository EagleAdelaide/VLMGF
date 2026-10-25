# FAR-Ground M2 Factor-Specific Operator Protocol

Frozen: 2026-07-15 before repair inference.

## Population

`data/processed/far_ground_m2_refcoco_val_200.jsonl` contains 200 unique RefCOCO val
images sampled from the matched M1 population using deployment-legal proxy strata.
Assigned disjoint quotas are:

| Stratum | n |
|---|---:|
| relation or spatial expression | 40 |
| likely part reference | 30 |
| small/medium predicted-area proxy | 35 |
| high cross-interface disagreement | 35 |
| low M1-router reliability proxy | 30 |
| easy proxy control | 30 |

Strata may overlap in eligibility, but each sample receives one assigned stratum. The
complete eligibility flags and proxy values are in `sample_manifest.csv`. GT IoU was
not used to assign samples.

## Frozen Operators

Qwen3-VL-4B uses tuple output in normalized 0-1000 coordinates because tuple is its M1
best fixed interface. InternVL3.5-4B uses its native ref-box output because that is its
M1 best fixed interface. Thus the repair instruction changes while each model retains
its strongest measured output convention.

### Relation-aware repair

The prompt explicitly requires resolving spatial/relational cues and distinguishing the
target from similar instances before returning one tight box.

### Part-aware repair

The prompt explicitly requires the smallest tight box around the referred visible part,
while retaining a tight whole-object box when the expression denotes the whole object.

Exact prompt strings are frozen in `src/prompting/grounding_prompts.py` under:

- `relation_repair_tuple_norm1000`
- `part_repair_tuple_norm1000`
- `internvl_relation_repair`
- `internvl_part_repair`

## Gates

Each operator is compared with that model's best fixed M1 interface on the identical
200 samples. A repair operator passes only if:

1. target-subset gain is at least `+0.03` mean IoU or `+0.05` IoU@0.5;
2. non-target mean-IoU degradation is no worse than `-0.01`;
3. paired bootstrap uncertainty is reported; and
4. the gain is not solely invalid-output recovery.

One-image and 20-sample gates precede the 200-sample run. No more than two GPU tasks run
concurrently.
