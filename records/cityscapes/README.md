# Cityscapes: supplementary Tables S5 and S6

This directory contains the preserved numerical summaries supporting the two
Cityscapes analyses in the supplement. The tables can be reconstructed on CPU
with Python 3.10 or later and the standard library:

```sh
python scripts/reproduce_cityscapes_tables.py --format json > cityscapes_checks.json
python scripts/reproduce_cityscapes_tables.py --write-tables ../cityscapes-tables > cityscapes_checks.csv
```

Run these commands from the repository root. `--records PATH` can select another
copy of this directory. A failed consistency or manuscript comparison exits with
status 1. CSV exports are written only when all checks pass. The reconstruction
uses the measurement summaries; `expected_tables.json` supplies independent
comparison targets, not the computed values.

| File | Contents |
|---|---|
| [table_S5.csv](table_S5.csv) | Table S5: eight budgets, with evaluation accuracy and latency for three families |
| [table_S6.csv](table_S6.csv) | Table S6: the primary oracle family for three seeds and two precisions |
| [shared_base_endpoints.csv](shared_base_endpoints.csv) | All 29 evaluation endpoints, at the precision preserved in the archive |
| [shared_base_budget_choices.csv](shared_base_budget_choices.csv) | All 32 frozen budget/family selections, including unavailable choices and the additional FixedBudget family |
| [shared_base_session_quantiles.csv](shared_base_session_quantiles.csv) | 87 session summaries: three p50/p95 pairs per endpoint |
| [shared_base_protocol.json](shared_base_protocol.json) | Table S5 data roles, timing protocol, model metadata and environment |
| [oracle_family_summaries.csv](oracle_family_summaries.csv) | All 54 archived family maxima and confidence intervals: nine families, three seeds, two precisions |
| [oracle_protocol.json](oracle_protocol.json) | Table S6 precision-specific environments, historical bootstrap procedure and interpretation |
| [expected_tables.json](expected_tables.json) | Displayed-value snapshot from the manuscript's supplementary numeric definitions |
| [provenance.json](provenance.json) | Original artifact names and SHA-256 hashes, export transformations and release scope |

## Table S5: shared-base comparison

`GlobalResize`, `SparseRefine` and `RCLC` are the three table families; `RCLC`
identifies the local correction probe. The archived `FixedBudget` family is
retained as context but is not a column of Table S5. Weights and routes were
selected using their recorded development/calibration roles. The reconstruction
preserves those choices; it does not choose a new winner using evaluation scores.

- `endpoint_id` joins the endpoint, choice and session files. Endpoint suffixes
  identify the archived resolution, threshold or workload setting.
- `budget_ms`, `development_p50_ms`, `evaluation_p50_ms`, `evaluation_p95_ms`,
  `p50_ms` and `p95_ms` are milliseconds. Budgets retain their original precision.
- Accuracy fields ending in `_percent` are pooled 19-class mIoU in percent.
  `calibration_miou` is a fraction in [0, 1], rather than a percentage.
- `status=selected` denotes a frozen choice. `no_feasible_candidate` has blank
  metric fields; the blank cells in `table_S5.csv` correspond to `--` in the paper.
- `evaluation_within_frozen_budget` compares evaluation p50 with the budget;
  feasibility for selection was determined from development timing.
- `session` is a zero-based timing-session index. Endpoint p50 and p95 each use
  the median of their three session quantiles. The script recomputes these medians.
  `historical_pooled_miou_percent` is the previously saved pooled-score check,
  repeated per session for the endpoint; it is not a session-specific accuracy.
- `quality_role` and `timing_role` identify the archived split roles.
  `n_model_replicas` is retained archive metadata, not the number of timing sessions
  or a basis for uncertainty estimates.

The quality evaluation uses 300 images, 19 classes and a 1024 x 2048 canvas.
Timing uses 50 images, three sessions and three repeats per session, with FP32
inference on the recorded RTX 5090 device. Table S5 prints accuracy to two decimal
places and budgets/latencies to three. Per-call timings and per-image confusion
arrays are not included in this deposit, so raw quantiles and pooled mIoU are not
recomputed from individual observations here.

## Table S6: label-informed diagnostic

The table extracts `family=oracle_abs_iou` from the family summaries, in FP32 then
FP16 order and with head seeds 812, 2027 and 3407. These heads share a backbone and
data. The oracle uses labels to select corrections and is not a deployable policy.

- `local_miou_percent` and `matched_linear_global_miou_percent` are pooled mIoU
  percentages. The latter is the global reference interpolated at the local
  expected mean latency.
- `maximum_margin_pp` is local minus matched global accuracy in percentage points;
  the script independently recalculates this subtraction for every family row.
- `fitted_expected_mean_latency_ms` is a fitted workload-dependent mean in
  milliseconds. `mean_k` is the mean selected correction workload, and
  `searched_family_candidates` is the number of candidate settings searched.
- `ci95_low_pp` and `ci95_high_pp` are the preserved 95% confidence limits in
  percentage points. `prediction_role` identifies the archived diagnostic role.
- `source_record_id` is a portable seed/precision identifier replacing private
  filesystem paths in the original CSV.

The historical intervals used 10,000 paired-image bootstrap draws, rebuilding the
global reference and reselecting the maximum within each frozen family. Latency
fit coefficients were held fixed; repeat-timing uncertainty is excluded. The
interval endpoints are retained from that analysis. This reconstruction does
not rerun the bootstrap, fit the latency model or rebuild predictions from images.
Table S6 prints all metrics to three decimal places.

S6 uses a different timing experiment from S5: its global reference uses measured
mean latency and its local workload uses fitted expected mean latency. These
results must not be pooled with S5's median-of-session-quantile measurements.

## Provenance and availability

The 14 November 2026 deposit exports archived endpoint, selection, timing-session
and oracle-family summaries. It adds no training, inference or bootstrap results
and changes no manuscript values. The original per-image arrays and per-call
timing arrays were not located for this deposit. Consequently, this release
supports reconstruction of the displayed tables and internal numerical checks,
but is not a complete raw-data reproduction package for the two experiments.

Cityscapes images, labels and model weights are not redistributed. Private paths
and complete device identifiers have been removed; environment metadata retains
device identifier prefixes to distinguish the experiments. The repository's
existing license applies to original study code and does not grant rights to
third-party datasets. See the root [license](../../LICENSE) and
[third-party terms](../../THIRD_PARTY_LICENSES.md).
