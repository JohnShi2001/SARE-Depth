# Reproduction commands and scope

## Frozen complete model

`python evaluate.py --device cpu --output outputs/example` runs the included two real RGB/depth pairs through the released base, DP-RHR and historical RE seed 20260711. `check_example.py` compares counts and metrics with the saved result. Two-sample assembly equivalence against the original checkpoint containers was checked on CPU with maximum absolute tensor difference zero at all three outputs. This directly verifies the tensor-container conversion and model assembly.

Full-set input uses `manifests/citrusfarm_test.csv` and a prepared dataset root as described in DATA.md. The evaluator exposes all three outputs and fails on invalid predictions on the fixed GT support.

## Round-2 controls

`experiments/round2_controls.py` is a path/container port of the actual executed control runner. `experiments/PORTABILITY_CHANGES.json` records its source hash and the specific portability changes. Numeric forward, objective, data conversion, training loop, checkpoint-selection and reduction logic are retained.

Linux shell example, after placing the prepared public pairs at the listed relative paths:

```bash
export SARE_DATA_ROOT=/path/to/prepared_citrusfarm
export SARE_EXPERIMENT_OUTPUT=/path/to/new_results_directory
export SARE_PUBLIC_WEIGHT=/path/to/depth_anything_v2_metric_vkitti_vits.pth
python experiments/round2_controls.py smoke
python experiments/round2_controls.py preprocessing
python experiments/round2_controls.py training
python experiments/round2_controls.py head_test
```

The first stage records the protocol and performs the bounded gradient/freeze test. Training comprises two one-epoch learning-rate pilots and three independent 30-epoch full-DPTHead runs. It requires an appropriate CUDA device. Every run starts from the released Stage-I base. Existing result directories are protected from replacement. These commands initiate new computation only when the reader explicitly runs them.

The public initializer is the official Depth Anything V2 Metric VKITTI Small checkpoint:
https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small

Expected upstream filename: `depth_anything_v2_metric_vkitti_vits.pth`.
Expected SHA-256: `9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`.
The control runner verifies this hash and sets `max_depth=80` explicitly.

The common square control uses PIL RGB bilinear interpolation. A separate legacy bridge retains the external-baseline OpenCV INTER_AREA square path and the historical reference precision. Refer to the per-configuration recorded results when reconstructing the manuscript's comparison; input interpolation and reference precision are part of its identity.

`python experiments/supplementary_controls.py public-area` evaluates the recovered public-checkpoint INTER_AREA square path on the original-FP32 reference grid. `python experiments/supplementary_controls.py matched-reference` evaluates the retained base, the two previously training-fitted scale coefficients and DP-RHR on that same grid. Both use the `SARE_DATA_ROOT`, `SARE_EXPERIMENT_OUTPUT` and `SARE_PUBLIC_WEIGHT` variables above. They create new, non-overwriting output files and perform inference only. The historical cached-reference bridge is preserved as a recorded result; the old FP16 cache is not bundled.

For inference with a selected completed head rather than new training:

```bash
python download_weights.py --include-heads
python evaluate.py --head-only --seed 20260711 --manifest manifests/citrusfarm_test.csv --data-root /path/to/prepared_citrusfarm --device cuda --output outputs/head_20260711
```

Repeat with seeds 20260712 and 20260713 in distinct output directories to evaluate the other retained heads. The complete metric-depth head replaces the base's original head; DP-RHR and RE are absent from this control.

## Distinct evidence cohorts

- Historical RE: three retained heads in the weight manifest, one frozen DP-RHR anchor, main-split training and the original evaluation conventions.
- New residual controls: the separate matched P2-A/P2-B/RE cohort in the manuscript, not replacements for the historical heads.
- Round-2 preprocessing: fixed public and study-base weights with paired input preprocessing.
- Round-2 head-only: three newly trained complete metric-depth heads, with validation selection and a separate test evaluation.
- Deployment: the manuscript's retained Jetson snapshots, backend tests and timings. Engines are not included in this minimal release.

The Stage-I checkpoint permits inference and adaptation from the retained model. Its original teacher-cache construction and continuation lineage are incompletely documented. The repository does not represent a from-scratch reconstruction of that historical training stage.
