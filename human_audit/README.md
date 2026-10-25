# Manual review of converted / external populations

Supports the manual-review sentence in Sec. 4 and Limitation (2) in Sec. 8.

## Design

- Population: 300 rows sampled from the external evaluation populations — 100 Visual7W,
  100 GQA (converted), 50 Visual Genome, 50 Flickr30k Entities.
- Blinding: the annotator saw only the query, the image, and the ground-truth box; no
  model predictions were shown.
- One annotator completed all 300 rows. No second annotator and no adjudication; the
  `annotator_2_*` and `adjudicated_*` columns are therefore empty, and agreement
  statistics are not computed.
- Labels: exactly one per row from the seven categories in `codebook.md`. All seven,
  including `valid_mildly_ambiguous`, were selectable in the annotation interface.

## Recorded labels (`labels_single_annotator.csv`; by family: `ingestion/label_counts_by_family.csv`)

| Category | Label(s) | Rows |
|---|---|---|
| Clearly valid | `valid_unambiguous` | 278 |
| Mildly ambiguous | `valid_mildly_ambiguous` | 0 |
| Invalid | `incorrect_gt_box` 9, `duplicate_near_duplicate` 5, `corrupted_image_query` 5, `invalid_target_query_conversion` 1 | 20 |
| Cannot determine | `cannot_determine` | 2 |
| Total | | 300 |

Every row carries an explicit label; there are no missing values. Free-text reasons
were optional and appear on seven rows. All seven mention other similar objects in the
image; they carry the labels `duplicate_near_duplicate` (3), `cannot_determine` (2),
`incorrect_gt_box` (1), and `valid_unambiguous` (1). Counts above are the labels as
recorded; no row was relabeled.

## Rates and exclusion rules (`ingestion/audit_summary.json`)

- Invalid = conversion error, incorrect ground-truth box, corrupted image/query, or
  duplicate. Unresolved = `cannot_determine`.
- Primary sensitivity excludes invalid and unresolved rows (22 rows). Mildly ambiguous
  rows would be retained because they are valid.
- Strict sensitivity additionally excludes mildly ambiguous rows; with zero such rows it
  coincides with the primary sensitivity.
- Excluded sample ids: `ingestion/filtered_sample_ids.csv`.

## Sensitivity results (`sensitivity/`)

Recomputed after the primary exclusion: per-family localization metrics and rankings
for the nine models (`localization_sensitivity.csv`, `family_model_ranks_*.csv`,
`rank_correlation_*.csv`) and deployment-refit GRC AUROC/AURC for the seven primary-grid
models on the four audited families (`grc_sensitivity.csv`; image-grouped five-fold,
seed 20260713).

- No family leader changes; maximum absolute rank-correlation change 0.0.
- Maximum absolute GRC change: AUROC 0.0041, AURC 0.0039.

This analysis was run under the parse protocol in place before the revision's parser
update; that update changed 15 analysis-population records and no success labels
(`data_compact/p0_4_reparse_audit_summary.json`).
