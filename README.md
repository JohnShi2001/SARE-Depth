# SARE-Depth

Research code for **Monocular Metric-Depth Adaptation for Agricultural Perception: A CitrusFarm Case Study with Embedded Inference**.

SARE-Depth keeps a teacher-guided Depth Anything V2 Small depth path frozen. DP-RHR learns image-level scale and gated local corrections. Four feature residuals then refine the frozen anchor through fixed depth-conditioned routing.

## Release contents

The `r2-20261103` research release contains the retained Stage-I checkpoint, the CitrusFarm DP-RHR adapter, three historical CitrusFarm RE checkpoints, three Round-2 complete metric-depth heads, inference and evaluation code, model configuration, and relative train/validation/test manifests. Weight files are Release assets, verified by size and SHA-256 using `configs/weights.json`.

The supplied Stage-I checkpoint is the study's trained base, not the public DAV2 initializer. The historical RE checkpoints correspond to the original three-run cohort. The later matched single-head experiment is a separate cohort.

## Quick start

Tested environment: Python 3.11, PyTorch 2.5.1, torchvision 0.20.1, NumPy 1.26.4, Pillow 10.4.0 and OpenCV 4.10.0.84. CPU execution is sufficient for the included example. A compatible PyTorch installation must be selected for the reader's device; CUDA execution in the recorded experiments used CUDA 12.4.

```bash
python -m venv .venv
# Activate this environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
python download_weights.py
python evaluate.py --device cpu --output outputs/example
python check_example.py outputs/example/summary.json
```

The two included RGB/depth pairs are actual CitrusFarm test samples. Their source names, recording files and timestamps are preserved in `manifests/example.csv`. `results/example_cpu_expected.json` records their measured CPU outputs. On the verified CPU environment, the distributed tensor-only weights and historical checkpoint containers produced identical base, anchor and final depth tensors for both samples. The example is a functional check, not an estimate of test-set performance.

Single-image inference produces an NPZ with `base`, `anchor` and `final` predictions on the 518 by 518 grid:

```bash
python infer.py example_data/images_v2/02_13B_Jackal_zed_2023_07_18_14_58_05_14_000217.png --output outputs/prediction.npz
```

Use `python download_weights.py --all-seeds` to fetch all historical RE heads. The final stage is changed with `evaluate.py --seed 20260712` or `--seed 20260713`.

The separately fine-tuned complete metric-depth head uses the same frozen base encoder:

```bash
python download_weights.py --include-heads
python evaluate.py --head-only --seed 20260711 --device cpu --output outputs/head_example
```

The `--seed` option selects one of the three retained head checkpoints. Evaluation performs no training or scale fitting.

## Data and evaluation

See [DATA.md](DATA.md) for the official source and the relative file layout. The complete manifests contain 3,500 training, 750 validation and 750 internal-test members. Source-bag membership is retained, including the original overlap across partitions. Full public source data are obtained from the original provider; this repository includes only two attributed examples.

```bash
python evaluate.py --manifest manifests/citrusfarm_test.csv --data-root /path/to/prepared_citrusfarm --device cuda --output outputs/full_test
```

The released evaluator uses FP32 original reference arrays, a fixed GT-defined finite support in [0.3, 30] metres, common cap30 and pooled-pixel metrics. Non-finite or non-positive predictions on that fixed support cause an error. Historical tables also contain FP16 cached-reference evaluations and uncapped model readouts. Match the protocol and checkpoint cohort when comparing a number with the manuscript.

## Reproduction scope

The release fixes the retained model and its forward computation, sample membership, preprocessing and evaluation. The original teacher-cache geometry and full continuation-training lineage remain incompletely documented. Starting from the retained base is the reproducible entry point for the released adaptation controls. This package does not include the teacher checkpoint, every historical fold model, or Jetson engines.

Round-2 preprocessing and complete metric-head controls are documented in `experiments/` with their locked settings and recorded results. These controls are distinct from the historical RE cohort. See [REPRODUCTION.md](REPRODUCTION.md) for exact commands and verification scope.

## Attribution and terms

Depth Anything V2 Small and vendored dependencies retain their original notices; see `licenses/DepthAnythingV2_LICENSE` and source headers. The original Depth Pro license is included as a provenance reference for the offline teacher. Depth Pro software or weights are not distributed here.

The two CitrusFarm examples are attributed to Hanzhe Teng, Yipeng Wang, Xiaoao Song and Konstantinos Karydis, *Multimodal Dataset for Localization, Mapping and Crop Monitoring in Citrus Tree Farms*, ISVC 2023. The dataset is [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The examples are extracted RGB PNGs and registered depth arrays in metres, with source identities retained. These extracted data remain CC BY-SA 4.0. See [DATA.md](DATA.md).

Study-specific code and checkpoint files are provided for inspection and reproduction. A repository-wide license has not been assigned; third-party materials retain their respective terms. No license for third-party data is superseded by this release.

Correspondence: Zhuoyu Shi, 1073324020194@st.gsau.edu.cn.
