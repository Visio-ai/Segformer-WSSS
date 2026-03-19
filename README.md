## Weakly Supervised Semantic Segmentation for Fast-Food Quality Control (SegFormer WSSS)

This repository contains the implementation used in the paper submission:

> **Weakly Supervised Semantic Segmentation for Fast-Food Quality Control: A Dual-Student Approach with SegFormer**

The method combines:

- a SegFormer backbone for classification and segmentation;
- a dual-student training strategy;
- pseudo-supervision from CAMs and TTA;
- a unified local dataset format.

## Quick start

```bash
uv sync
uv run python train.py --config configs/mini.json --dataset-root datasets/WeakSupervisionSegmentation-mini:v1
```

Optional public-data flow:

```bash
uv run python scripts/convert_foodseg103.py --output-dir datasets/FoodSeg103_converted
uv run python train.py --config configs/default.json --dataset-root datasets/FoodSeg103_converted
```


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


## Limitations

- Main dataset used in the paper is private.
- Included mini dataset is for reproducibility checks, not for benchmarking final performance.
- Repository layout may still change as monorepo alignment progresses.

## Citation

If you use this repository in academic work, please cite the corresponding paper. A BibTeX entry will be added once publication metadata is available.