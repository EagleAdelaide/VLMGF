# Converted-Dataset Human Audit Codebook

Assign exactly one label per row.

| Label | Use when |
|---|---|
| `valid_unambiguous` | The query clearly denotes one target and the green box correctly encloses it. |
| `valid_mildly_ambiguous` | The intended target is plausible, but wording or multiple similar candidates creates limited ambiguity. |
| `invalid_target_query_conversion` | The converted query does not denote the boxed target or the conversion changes the intended task. |
| `incorrect_gt_box` | The query is meaningful, but the green box is materially misplaced, incomplete, or encloses the wrong object. |
| `corrupted_image_query` | The image is unreadable/missing or the query text is corrupted. |
| `duplicate_near_duplicate` | The row duplicates or nearly duplicates another audited item in a way that compromises the sample audit. |
| `cannot_determine` | Available visual or linguistic evidence is insufficient for any other label. |

## Boundaries

- A tight box around a visible object is valid even if it excludes contextual objects.
- Minor annotation looseness is not an incorrect box unless it changes target identity or
  materially changes localization.
- Mere query brevity is not ambiguity when only one matching target exists.
- Multiple candidates are mildly ambiguous only when the expression does not resolve
  them; do not infer an unstated relation.
- Record what is visible. Do not infer model behavior or desired paper outcomes.
