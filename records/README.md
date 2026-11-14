# Record structure

`main_timing/` contains the primary DeepGlobe/Inria benchmark, full confusion
matrices, per-image IoUs, raw per-call timings, quantiles, specialized-method
memory and separate CPU/GPU partitions. `replication_timing/` contains historical
reproduction/replication observations separately; its `calls.csv` is the raw
replication timing benchmark. `quality/` contains fixed validation ranks,
selected-seed total confusion matrices and additional-seed validation/test trees.
`urur/` contains aggregate route metrics and total confusion matrices only;
per-call `image` values are anonymous ordinal positions.

`cityscapes/` contains preserved endpoint, budget-selection, session-quantile and
oracle-family summaries for supplementary Tables S5 and S6, along with the two
rendered CSV tables. Run `scripts/reproduce_cityscapes_tables.py` for their separate
reconstruction and consistency checks. Original per-image/per-call arrays are not
included for these experiments, and historical bootstrap intervals are retained
without rerunning resampling. See [the Cityscapes record guide](cityscapes/README.md)
for units, provenance and the distinction between its two timing protocols.

Status fields distinguish complete, unavailable and unrequested observations.
URUR R18 has no timing measurement; its full-resolution route was unavailable.
Published URUR ISDNet accuracy is distinct from measured retrained accuracy.
Published literature scores are inputs with their own original protocols.

`manuscript/expected_tables.json` is the independent displayed-value snapshot.
It is used only for comparisons. The computation uses confusion matrices,
raw calls, validation ranks, publication anchors and seed records. The release contains the numerical evidence and its reconstruction code.

Benchmark labels: R4 denotes the primary DeepGlobe/Inria benchmark, R3 its
replication benchmark, and R5-T the dedicated URUR benchmark.
