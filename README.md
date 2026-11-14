# When Does Specialization Pay Off in Ultra-High-Resolution Remote Sensing Segmentation?

**Auditing Baselines, Metrics, and End-to-End Efficiency**  
**Keyao Ma** · Manuscript under review

This repository contains the evaluation routes, executed configurations,
measurement records and analysis for controlled UHR segmentation comparisons on
DeepGlobe, Inria and URUR. Accuracy is computed from pooled confusion matrices.
Generic routes are selected through validation ranks, with optional latency
budgets. The empirical upper hull interpolates in physical milliseconds and
ends at the best observed generic accuracy.

## Reproduce the numbers on CPU

Python 3.10 or later is sufficient; numeric reproduction uses only the standard
library. Create the environment outside the repository:

```sh
python -m venv ../uhr-records-venv
# Linux/macOS:
../uhr-records-venv/bin/python -m pip --isolated --disable-pip-version-check install --no-index -r environment/cpu-records.txt
../uhr-records-venv/bin/python scripts/reproduce_tables.py > comparison.csv
# Windows PowerShell:
../uhr-records-venv/Scripts/python.exe -m pip --isolated --disable-pip-version-check install --no-index -r environment/cpu-records.txt
../uhr-records-venv/Scripts/python.exe scripts/reproduce_tables.py > comparison.csv
```

The script reads only `records/`, recomputes accuracy, raw-call means and
quantiles, memory, budget selections, recovery and seed statistics, then compares
the rendered values in Tables II–VII and the Figure 3 coordinates. It exits with
status 1 on any mismatch. `--format json` includes the full comparison and plot
coordinates. The manuscript snapshot supplies the comparison values.

Supplementary Cityscapes Tables S5 and S6 have a separate reconstruction:

```sh
python scripts/reproduce_cityscapes_tables.py --format json > cityscapes_checks.json
```

The [Cityscapes records](records/cityscapes/README.md) include full-precision
archived summaries and viewable CSV tables. The script reconstructs the table
rows, checks session-quantile aggregation and accuracy differences, and compares
the displayed values with the manuscript. Per-image arrays and original per-call
timings for these two experiments are not included; confidence bounds are retained
from the historical bootstrap rather than recomputed by this script.

## Model environments

The main timing environment was Python 3.10.20, PyTorch 2.0.0+cu118,
MMCV-full 1.7.2 and MMSegmentation 0.30.0. Dedicated URUR timing used Python
3.10.22 with the same framework versions. `environment/` contains the recorded
variants, the main full package freeze and the specialized-method additions.
Historical replication and main timing remain separate.

Example Linux installation for model execution:

```sh
conda env create -f environment/main-timing.yml
conda activate uhr-main
python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
python -m pip install -r environment/fctl-extra.txt
```

Use `environment/urur-timing.yml` for the dedicated URUR environment. External
model code is pinned below; install its dependencies without replacing these
framework versions. The CPU environment reproduces the recorded analyses. Model execution uses
the corresponding CUDA environment.

```sh
git clone https://github.com/cedricgsh/ISDNet.git external/ISDNet
git -C external/ISDNet checkout 7341335677fd6201457b321e1e6368feb7ad647e
git clone https://github.com/liqiokkk/FCtL.git external/FCtL
git -C external/FCtL checkout 6fdb0c02f22efbf267781857c9ee7f7ebc951974
git clone https://github.com/VinAIResearch/MagNet.git external/MagNet
git -C external/MagNet checkout 7f5b8760cb471230402015d80117559de0e38600
```

ISDNet's pinned repository has an Apache-2.0 LICENSE. Its model implementation
is loaded from that external checkout. No MMSegmentation library source is
vendored. Obtain checkpoints and initializers from their
providers and stored outside the tracked repository.

## Data

Obtain [DeepGlobe](https://deepglobe.org/challenge.html) and
[Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/) from
their official providers. This study uses DeepGlobe 454/207/142 and Inria
126/27/27 train/validation/test splits; the lists are in `splits/`. These are
study splits within publicly supplied labeled data, separate from hidden
challenge test servers. URUR is **available from its authors under their terms**.
Request its authorized splits from the authors; no URUR identifiers are included.

For training, arrange indexed labels and images under a local dataset root as
`images/<id>.<suffix>` and `labels/<id>.png`; `configs/README.md` specifies label
indices. Set the image suffix in the local configuration to match the prepared
files. For evaluation, create an external JSON list of objects with `image` and
`label` paths relative to the manifest. These manifests are local inputs, not
release artifacts. Raw DeepGlobe RGB and Inria binary labels are supported by
the corresponding `--label-encoding` argument.

## Commands

Training examples (use the corresponding configuration for other models/seeds):

```sh
python -m scripts.train --config configs/deepglobe/b5_seed0.json --data-root ../datasets/deepglobe --split splits/deepglobe/train.txt --output ../runs/dg_b5_seed0
python -m scripts.train --config configs/inria/r50_historical.json --data-root ../datasets/inria --split splits/inria/train.txt --output ../runs/inria_r50
python -m scripts.train --config configs/urur/isdnet.json --data-root ../datasets/urur --split ../private_splits/urur_train.txt --isdnet-repo external/ISDNet --shallow-weight ../weights/STDCNet813M_73.91.tar --output ../runs/urur_isdnet
```

The measured DG/Inria R50 seed is 20260929. B5 has measured seeds 0/1/2. URUR ISDNet uses microbatch 2, accumulation 4 and
160000 updates; its measured accuracy remains distinct from the published
reference used in Table VII and Figure 3.

```sh
python -m scripts.evaluate --config configs/deepglobe/b5_seed0.json --checkpoint ../weights/dg_b5_seed0.pth --manifest ../manifests/dg_test.json --label-encoding deepglobe_rgb --output ../runs/dg_test.json
python -m scripts.evaluate --config configs/deepglobe/b5_seed0.json --checkpoint ../weights/dg_b5_seed0.pth --manifest ../manifests/dg_timing.json --time --output ../runs/dg_calls.csv
python -m scripts.isdnet_config --repo external/ISDNet --dataset deepglobe --output ../configs/isdnet_dg.json
python -m scripts.evaluate --config ../configs/isdnet_dg.json --checkpoint ../weights/isdnet_dg.pth --isdnet-repo external/ISDNet --manifest ../manifests/dg_test.json --label-encoding deepglobe_rgb --output ../runs/isdnet_dg.json
python -m scripts.specialized --factory adapters.fctl.endpoint:FCtLEndpoint --code external/FCtL --weights ../weights/fctl --receipt adapters/fctl/source_receipt.json --dataset inria --variant FCtL --manifest ../manifests/inria_timing.json --time --output ../runs/fctl_calls.csv
python -m scripts.specialized --factory adapters.magnet_agpl.endpoint:MagNetEndpoint --code external/MagNet --weights ../weights/magnet --receipt adapters/magnet_agpl/source_receipt.json --dataset deepglobe --variant MagNet-Fast --manifest ../manifests/dg_timing.json --time --output ../runs/magnet_fast_calls.csv
python scripts/reproduce_tables.py > comparison.csv
```

The timing manifests select 20 DeepGlobe or 10 Inria test images using NumPy
`default_rng(20260930)`, without replacement, with sorted split positions.
URUR uses its dedicated ten-image timing subset. Use only that benchmark's
authorized external list. The timed calls are anonymized by list position.
OOM routes are unavailable; substituting a sliding route changes the experiment.

## Timing boundary and hardware

The measured boundary matches Section IV-C: decoded CPU RGB uint8 input to a
full-resolution GPU class map, with CUDA synchronization at both endpoints.
It includes host-to-device transfer, normalization, resizing, network execution,
reconstruction and argmax; disk reads and image decoding are excluded.
ISDNet official preprocessing uses CPU MMCV/OpenCV. Generic and ISDNet GPU
routes perform their prescribed preprocessing on GPU. FCtL retains CPU
aggregation, official TTA and the final transfer to GPU. MagNet retains the
released patch/refinement pipeline. DG/Inria ISDNet uses nominal scale 0.8;
URUR uses scale 1.0.

Main measurements use NVIDIA GeForce RTX 4090, PyTorch intra-op/inter-op threads
4/1, OpenCV threads 1, cuDNN benchmarking enabled and TF32 disabled. Batch size
is 1, with two warm-up passes and five repeats. Route order rotates by image and
repeat within each model family. Allocated memory includes resident model
allocations; reserved memory also includes allocator caching. Specialized memory
and CPU partition measurements have their own records. CPU shares are computed
from separate partition blocks, rather than dividing by complete-path times.

## Manuscript mapping

| Item | Inputs |
|---|---|
| Table I | `audit/table_I_evidence.json`, `audit/table_I_updated.csv` |
| Table II: reproduction | `records/main_timing/specialized.json`, `routes.json`, published references |
| Table III: baselines | `records/main_timing/fixed_input.json`, route confusion matrices |
| Table IV: recovery | primary route records, `records/quality/seed0_validation.json`, published references |
| Table V: efficiency | `records/main_timing/calls.csv`, `specialized_calls.csv`, `partitions.json`, specialized memory |
| Table VI: seeds | `records/quality/seed_metrics.json`, primary seed-0 route timings |
| Table VII: URUR | `records/urur/confusion_matrices.json`, `calls.csv`, published references |
| Figure 3: coordinates | primary and URUR route records, checked by `scripts/reproduce_tables.py` |
| Supplementary Table S5: Cityscapes shared-base comparison | [frozen choices, endpoint and session summaries](records/cityscapes/README.md#table-s5-shared-base-comparison) |
| Supplementary Table S6: Cityscapes oracle diagnostic | [archived family maxima and confidence intervals](records/cityscapes/README.md#table-s6-label-informed-diagnostic) |

Figure drawing sources are not included. No dataset pixels, labels, model
weights, or URUR identifier lists are distributed. Unmeasured routes and
unavailable measurements remain explicit in the records.

## License and citation

Original study code: MIT, Copyright (c) 2026 Keyao Ma. Third-party terms are listed
in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md); MagNet integration is AGPL-3.0.
`CITATION.cff` contains machine-readable citation metadata.

```bibtex
@unpublished{ma2026specialization,
  author = {Ma, Keyao},
  title = {When Does Specialization Pay Off in Ultra-High-Resolution Remote Sensing Segmentation? Auditing Baselines, Metrics, and End-to-End Efficiency},
  year = {2026},
  note = {Manuscript under review}
}
```
