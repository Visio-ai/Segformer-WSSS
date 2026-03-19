## Weakly Supervised Semantic Segmentation for Fast-Food Quality Control (SegFormer WSSS)

This repository contains the implementation used in the paper submission:

> **Weakly Supervised Semantic Segmentation for Fast-Food Quality Control: A Dual-Student Approach with SegFormer**

The method combines:

- a SegFormer backbone for classification and segmentation;
- a dual-student training strategy;
- pseudo-supervision from CAMs and TTA;
- a unified local dataset format.

## TL;DR (Quick start)

```bash
uv sync
uv run python train.py --config configs/mini.json --dataset-root datasets/WeakSupervisionSegmentation-mini:v1
```

Optional public-data flow:

```bash
uv run python scripts/convert_foodseg103.py --output-dir datasets/FoodSeg103_converted
uv run python train.py --config configs/default.json --dataset-root datasets/FoodSeg103_converted
```

## Highlights

- Reproducible training pipeline with PyTorch Lightning.
- Local dataset spec shared by mini and converted public datasets.
- Support for weak supervision from image-level labels and CAM-based pseudo-masks.
- End-to-end training entrypoint: `train.py`.

## Repository data policy

Some review comments asked whether training/validation-related folders should be versioned. For this repository:

- `datasets/WeakSupervisionSegmentation-mini:v1` is intentionally versioned to provide a ready-to-run reproducibility sample and is publicly shareable with authorization from the dataset owner.
- `VSS_stats/` is intentionally versioned because it contains only aggregate statistics/artifacts from the private VSS dataset (no private raw images or labels).
- Large or private datasets are not redistributed in this repo.

For a future monorepo migration, the recommended direction is to keep only minimal reproducibility assets in-repo and move larger datasets/artifacts to external storage with bootstrap scripts.

## Project structure

Current layout (incremental organization, no breaking refactor):

```text
.
├── train.py                        # main training entrypoint
├── tta.py                          # inference/TTA utilities
├── model.py                        # dual-student LightningModule
├── backbone.py                     # SegFormer multitask wrapper
├── dataset.py                      # dataset format + dataset classes
├── datamodule.py                   # Lightning DataModule
├── augmentation.py                 # CutMix/MixUp for multi-label WSSS
├── metrics.py                      # mIoU metric
├── callbacks.py                    # training callbacks
├── helpers/                        # transforms, samplers, CAM helpers, metrics
├── scripts/
│   └── convert_foodseg103.py       # converter for public FoodSeg103
├── configs/                        # experiment configs
├── datasets/                       # sample/public-format datasets
└── VSS_stats/                      # aggregate stats for private VSS dataset
```

Monorepo roadmap (planned, not applied in this PR): migrate core modules to `src/segformer_wsss/` and keep runnable entrypoints in `scripts/`.

## Setup

### Requirements

- Python `>=3.10`
- CUDA-capable GPU recommended
- `uv` for environment management

Install dependencies:

```bash
uv sync
```

## Method overview

The training pipeline is centered on `WSSSegmentationModel`:

1. Two students (`model1`, `model2`) learn jointly.
2. Early training emphasizes classification and discrepancy losses.
3. Later training uses pseudo-labels generated from CAMs and TTA, with cross-teaching.

## Datasets

### Private VSS dataset

The full VSS dataset (Visio Sandwich Segmentation) used in the paper is private and is not redistributed in this repository. This repository includes only aggregate artifacts in `VSS_stats/`.

### Included mini dataset

`datasets/WeakSupervisionSegmentation-mini:v1` is a small sample in the local format and is intended for fast smoke tests and reproducibility checks.

### Data availability and usage rights

The folder `datasets/WeakSupervisionSegmentation-mini:v1` is intentionally included as a publicly shareable mini version of the dataset, released with authorization from the dataset owner for research and reproducibility purposes.

The full VSS dataset remains private. The `VSS_stats/` directory contains only aggregate statistics and non-identifying artifacts.

### Public FoodSeg103 conversion

Use `scripts/convert_foodseg103.py` to convert FoodSeg103 into the local format used by this project.

## Training and evaluation commands

- Show training CLI options:

```bash
uv run python train.py --help
```

- Train with mini dataset:

```bash
uv run python train.py --config configs/mini.json --dataset-root datasets/WeakSupervisionSegmentation-mini:v1
```

- Convert and train with FoodSeg103:

```bash
uv run python scripts/convert_foodseg103.py --output-dir datasets/FoodSeg103_converted
uv run python train.py --config configs/default.json --dataset-root datasets/FoodSeg103_converted
```

## Reproducibility notes

- Core dependencies are managed via `pyproject.toml` and `uv.lock`.
- Prefer running all commands with `uv run`.
- For consistent runs, keep configs under `configs/` and log outputs/checkpoints outside source folders.

## Use of Generative AI

Generative AI tools were used as engineering assistance for tasks such as documentation drafting, structure cleanup suggestions, and boilerplate/code organization support.

All model design decisions, implementation choices, experiment setup, and final validation were reviewed and approved by the authors/maintainers.

## Hardware used for experiments

The main experiments and end-to-end tests for this repository were run on the same workstation that hosts this repo, with the following specifications:

- **Operating system**: Linux (kernel `6.14.0-37-generic`).
- **Python**: 3.10+ (matching the `>=3.10` requirement in `pyproject.toml`).
- **GPU**: NVIDIA GeForce RTX 5090 with **32 GB VRAM**.
- **CPU**: Intel Core i7‑14700KF.
- **RAM**: 32 GB system memory.

You can train smaller configurations (e.g. the mini dataset) on more modest hardware, but for full-resolution training on larger datasets like FoodSeg103 or the private VSS dataset, having a modern CUDA-capable GPU with ≥ 20 GB VRAM and at least 32 GB system RAM is strongly recommended.

## Limitations

- Main dataset used in the paper is private.
- Included mini dataset is for reproducibility checks, not for benchmarking final performance.
- Current repository layout is functional but still transitional pending monorepo alignment.

## Citation

If you use this repository in academic work, please cite the corresponding paper. A BibTeX entry will be added once publication metadata is available.
