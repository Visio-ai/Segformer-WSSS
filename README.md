## Weakly Supervised Semantic Segmentation for Fast-Food Quality Control (SegFormer WSSS)

This repository is an **organized version** of the codebase used for the submission of the paper:

> **Weakly Supervised Semantic Segmentation for Fast-Food Quality Control: A Dual-Student Approach with SegFormer**

It provides a reproducible implementation of **Weakly Supervised Semantic Segmentation (WSSS)** for fast‑food quality control, based on:

- A **SegFormer** backbone used for both classification and segmentation.
- A **dual-student** architecture (`WSSSegmentationModel` with `model1` and `model2`) sharing the backbone.
- Training with **image-level labels** and **pseudo-masks** derived from Class Activation Maps (CAMs) and Test-Time Augmentation (TTA).
- A **standard dataset format** shared by all datasets used in this project.

The code is built around a single training script (`train.py`), a local dataset specification (`dataset.yaml` + `annotations` + `splits`), and PyTorch Lightning modules for training and evaluation.

---

## Quick start

The fastest way to test the pipeline is to use the included mini sample dataset with `uv`.

1. **Create environment with `uv`**

   ```bash
   uv sync
   ```

2. **Run training on the mini sample dataset**

   ```bash
   uv run python train.py --config configs/mini.json --dataset-root datasets/WeakSupervisionSegmentation-mini:v1
   ```

3. **(Optional) Convert FoodSeg103 and train on it**

   ```bash
   # Convert FoodSeg103 to the local format
   uv run python scripts/convert_foodseg103.py --output-dir datasets/FoodSeg103_converted

   # Train on the converted FoodSeg103 dataset
   uv run python train.py --config configs/default.json --dataset-root datasets/FoodSeg103_converted
   ```

For more details about environment setup, dataset format, and model structure, see the sections below.

---

## Environment requirements

- **Python**:
  - The project requires **Python >= 3.10**, as specified in `pyproject.toml`.
- **GPU**:
  - A CUDA-capable GPU is strongly recommended.
  - Training uses PyTorch and PyTorch Lightning and, by default, lets Lightning pick devices:
    - `pl.Trainer(devices="auto")` in `train.py`.
- **Dependency management**:
  - The repository is designed to work with **[`uv`](https://astral.sh/docs/uv/)**, a fast replacement for `pip`/`venv`.
  - The files `pyproject.toml` and `uv.lock` describe the environment in a reproducible way.

### Installing `uv`

Please refer to the official `uv` documentation for your platform. On Unix-like systems, a typical installation command is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, the `uv` command should be available on your `PATH`.

---

## Project setup with `uv`

### 1. Clone the repository

```bash
git clone https://github.com/Visio-ai/Segformer-WSSS-Public.git
cd Segformer-WSSS-Public
```

### 2. Create the environment and install dependencies

Inside the project directory, run:

```bash
uv sync
```

This will:

- Create (if needed) a virtual environment managed by `uv`.
- Install all dependencies specified in `pyproject.toml` / `uv.lock`.

To run Python scripts inside this environment, use `uv run`:

```bash
uv run python train.py --help
```

### 3. Alternative without `uv` (pip/venv)

If you do not want to use `uv`, you can:

1. Create and activate a virtual environment (for example, `python -m venv .venv` and `source .venv/bin/activate`).
2. Manually install the main dependencies, e.g.:

```bash
pip install torch torchvision
pip install pytorch-lightning mlflow transformers
pip install datasets huggingface_hub pyyaml opencv-python pillow numpy
```

Make sure to install any other dependencies used in the code, as listed in `pyproject.toml` if you extend it.

---

## Repository structure (high level)

Main files and directories:

- `train.py`: main WSSS training script (PyTorch Lightning).
- `model.py`: defines `WSSSegmentationModel`, the dual-student LightningModule.
- `backbone.py`: `SegformerMultiTask` wrapper around SegFormer for classification + segmentation + CAMs.
- `dataset.py`: local dataset format definition and train/val Dataset implementations.
- `datamodule.py`: `WSSDataModule` (Lightning DataModule) that builds the train/val DataLoaders.
- `augmentation.py`: CutMix and MixUp tailored for multi-label WSSS.
- `callbacks.py`: callbacks for memory logging and per-epoch checkpoint saving.
- `metrics.py`: mIoU metric implementation for semantic segmentation.
- `helpers/`: utilities (transforms, Grad-CAM, samplers, additional metrics, etc.).
- `scripts/convert_foodseg103.py`: script to convert the **FoodSeg103** dataset (HuggingFace) to the local format.
- `configs/`:
  - `default.json`: default experiment configuration.
  - `mini.json`: lightweight configuration for the mini sample dataset.
- `datasets/WeakSupervisionSegmentation-mini:v1`: **sample dataset** already in the expected local format.
- `datasets/FoodSeg103_converted/`: example path for the converted FoodSeg103 dataset (created by the script).
- `VSS_stats/`: statistics and artifacts for the **private dataset** used in the paper.

---

## Datasets

### Private paper dataset

The real fast-food dataset Visio Sandwich Segmentation (VSS) used in the paper is **private** and cannot be redistributed.

- The repository includes only aggregate statistics and related artifacts in:
  - `VSS_stats/`

### Sample dataset (mini) of VSS

To make the pipeline easy to test, we provide a **sample dataset** that already follows the local format:

- Directory: `datasets/WeakSupervisionSegmentation-mini:v1`
- Structure:
  - `dataset.yaml`: dataset metadata (number of classes, class names, label formats, stats, etc.).
  - `train/images/`: training images.
  - `val/images/`: validation images.
  - `val/labels/`: YOLO-polygon label files (`.txt`) and cached masks (`.png`) when applicable.
  - `annotations/train_labels.json`: image-level multi-hot labels for training.
  - `annotations/val_labels.json`: image-level multi-hot labels for validation.
  - `splits/single/train.txt` and `splits/single/val.txt`: training/validation file lists.

You can use this directory directly as `--dataset-root` when running `train.py`.

### FoodSeg103 (public dataset)

The **FoodSeg103** dataset (available on the HuggingFace Hub) is used as an example of a public dataset that can be converted to the local format.

The script `scripts/convert_foodseg103.py`:

- Downloads the dataset from HuggingFace (default: `EduardoPacheco/FoodSeg103`).
- Converts per-pixel segmentation masks into:
  - **Image-level multi-hot labels** (classification).
  - **YOLO polygon labels** (`.txt`) with a **mask cache** (`.png`) where pixels store class IDs.
- Generates:
  - `annotations/train_labels.json` and `annotations/val_labels.json`
  - `splits/single/train.txt` and `splits/single/val.txt`
  - `dataset.yaml` describing classes, label formats, and basic stats.

After conversion, the output directory (e.g. `datasets/FoodSeg103_converted/`) is ready to be used as `--dataset-root` in training.

---

## Local dataset format

Every dataset compatible with this repository must follow a common structure in its root directory:

- `dataset.yaml`:
  - Defines:
    - `num_classes` and `num_foreground_classes`.
    - `class_names` (id → name mapping).
    - Label formats for classification and segmentation.
    - Cross-validation information (`splits/single` or `splits/{k}`).
- Typical directory layout:
  - `train/images/`: training images.
  - `val/images/`: validation images.
  - `val/labels/`:
    - YOLO polygon label files (`.txt`).
    - Cached segmentation masks (`.png`) where pixel values are class IDs.
  - `annotations/train_labels.json`: foreground-only multi-hot vectors per training image.
  - `annotations/val_labels.json`: foreground-only multi-hot vectors per validation image.
  - `splits/single/train.txt`: list of base names (without extension) for training images.
  - `splits/single/val.txt`: list of base names for validation images.

Both the sample dataset `WeakSupervisionSegmentation-mini:v1` and the converted FoodSeg103 dataset follow this specification.

---

## Generating the FoodSeg103 dataset in the local format

### 1. Extra dependencies

In addition to the core project dependencies, the conversion script uses:

- `datasets`
- `huggingface_hub`
- `pyyaml`
- `opencv-python`
- `pillow`
- `numpy`

If you ran `uv sync`, these should already be available. Otherwise, you can install them manually:

```bash
pip install datasets huggingface_hub pyyaml opencv-python pillow numpy
```

### 2. Run the converter

To convert FoodSeg103 into the local format, run:

```bash
uv run python scripts/convert_foodseg103.py --output-dir datasets/FoodSeg103_converted
```

Main arguments:

- `--output-dir`: root directory where the converted dataset will be created.
- `--dataset-name` (optional): dataset name on HuggingFace (default: `EduardoPacheco/FoodSeg103`).
- `--cache-dir` (optional): cache directory to use with `datasets`.

### 3. Generated structure

After the script finishes, you should see something like:

- `datasets/FoodSeg103_converted/`
  - `train/images/`
  - `val/images/`
  - `val/labels/`:
    - `val_00000.txt`, `val_00000.png`, etc.
  - `annotations/`:
    - `train_labels.json`
    - `val_labels.json`
  - `splits/single/`:
    - `train.txt`
    - `val.txt`
    - `stats.json`
  - `dataset.yaml`

This directory can be passed directly as `--dataset-root` to `train.py`.

---

## Training

The main entry point for training is `train.py`. It uses:

- A JSON configuration file (`--config`).
- A dataset root directory (`--dataset-root`) following the local format.
- Optionally, a JSON file with MLflow credentials (`--mlflow-creds`).

### Example 1: quick run with the sample mini dataset

Recommended first run to verify the pipeline:

```bash
uv run python train.py \
  --config configs/mini.json \
  --dataset-root datasets/WeakSupervisionSegmentation-mini:v1
```

In this example:

- `configs/mini.json` contains a lightweight configuration (e.g. fewer epochs) suitable for the mini dataset.
- `--dataset-root` points to the included sample dataset.

### Example 2: training with converted FoodSeg103

After generating the FoodSeg103 dataset:

```bash
uv run python train.py \
  --config configs/default.json \
  --dataset-root datasets/FoodSeg103_converted
```

### Key configuration parameters

The JSON config files under `configs/` control most aspects of training. Some important fields:

- **Data and training**:
  - `dataset_root`: can be overridden by `--dataset-root`.
  - `epochs`: number of training epochs.
  - `batch_size`: batch size.
  - `image_size`: image resolution (defaults to 448 in the code if not set).
  - `num_workers`: number of DataLoader workers.
  - `use_balanced_sampler`: whether to use a multi-label balanced sampler.
  - `sampler.num_samples`: number of samples per epoch for the sampler.
- **Model and losses**:
  - `backbone_model`: SegFormer model name (e.g. `nvidia/mit-b0`).
  - `freeze_backbone`: if `true`, freeze the backbone and train only heads.
  - `lambda_cls`, `lambda_seg`, `lambda_dis`: loss weights for classification, segmentation, and discrepancy.
  - `background_weight`, `foreground_weight`: class weights for the segmentation loss.
- **Training flow**:
  - `stage_transition_pct`: fraction of epochs for stage 0 (classification + discrepancy) before switching to stage 2 (pseudo-mask training).
  - `tta`: options controlling pseudo-label generation with Test-Time Augmentation.
- **Logging and checkpoints**:
  - `checkpoint_dir`: where checkpoints are saved.
  - `debug`: if `true`, limits the number of batches (useful for debugging).
  - `enable_profiling`: enables profiling callbacks and memory logging.
- **MLflow (optional)**:
  - `online`: if `true`, enable MLflow logging (if credentials are available).
  - `mlflow_creds_path`: path to a JSON file with:
    - `MLFLOW_TRACKING_URI`
    - `MLFLOW_TRACKING_USERNAME`
    - `MLFLOW_TRACKING_PASSWORD`

If `online` is `true` but the credentials file cannot be loaded, `train.py` will automatically disable MLflow logging and continue with CSV-only logging.

---

## What each main code file contains

### `train.py`

- Main training script.
- Parses CLI arguments:
  - `--config`: path to a JSON config file (default: `configs/default.json`).
  - `--dataset-root`: dataset root directory (overrides `dataset_root` in the config).
  - `--mlflow-creds`: path to a JSON file with MLflow credentials.
- Loads the config, sets random seeds and matmul precision.
- Builds:
  - `WSSSegmentationModel` (dual-student SegFormer-based model).
  - Class weights for segmentation.
  - Loggers: CSV logger and, optionally, an MLflow logger.
  - Checkpoint and profiling callbacks (including memory logging).
  - `WSSDataModule` with all dataset-related options.
- Launches training with `trainer.fit(model, datamodule)`.

### `scripts/convert_foodseg103.py`

- Converts the HuggingFace FoodSeg103 dataset to the local format.
- Downloads the dataset (default: `EduardoPacheco/FoodSeg103`).
- For each sample:
  - Saves RGB images as `.jpg` under `train/images/` or `val/images/`.
  - Converts segmentation masks into:
    - Multi-hot vectors for classification (excluding background).
    - YOLO polygon labels (`.txt`) and cached masks (`.png`) for validation.
- Generates:
  - `annotations/train_labels.json` and `annotations/val_labels.json`.
  - `splits/single/train.txt` and `splits/single/val.txt`.
  - `dataset.yaml` describing the dataset, label formats, and stats.
- Requires: `datasets`, `huggingface_hub`, `pyyaml`, `opencv-python`, `pillow`, `numpy`.

### `dataset.py`

- **`DatasetConfig`**:
  - Loads and validates `dataset.yaml`.
  - Exposes:
    - `num_classes`, `num_foreground_classes`.
    - `class_names` dictionary.
    - Paths for train/val images, labels, annotations, and splits.
  - Provides helpers for:
    - Single split (`splits/single/train.txt`, `annotations/train_labels.json`).
    - K-fold splits (`splits/{n_folds}/fold_{fold}/...`).
- **YOLO mask utilities**:
  - `load_yolo_masks`: builds per-class masks from YOLO polygon labels.
  - `generate_mask_cache`: creates cached `.png` masks from the `.txt` files (atomic writes, multi-worker safe).
  - `load_mask_from_cache_or_txt`: loads masks from cache or recomputes them as needed.
- **`WSSTrainDataset`**:
  - Training dataset based on `train.txt` + `train_labels.json`.
  - Applies:
    - `MaskAwareTransforms` (data augmentation + valid-region mask).
    - Optional `DomainCropAugmentation` for domain-specific cropping/obscuring.
  - Returns tuples: `(image, label_with_bg, bg_mask)`.
  - Provides `get_labels()` for use with the balanced sampler.
- **`WSSValDataset`**:
  - Validation dataset using `val/images/` and `val/labels/`.
  - Loads YOLO polygon labels + cached masks and builds per-class masks.
  - Applies `MaskAwareValidationTransforms` to images.
  - Returns: `(image, label_with_bg, masks[num_classes, H, W], idx)`.

### `datamodule.py`

- Defines **`WSSDataModule`**, a PyTorch Lightning `LightningDataModule`.
- Responsibilities:
  - Creates a `DatasetConfig` for the selected `dataset_root`.
  - Builds `WSSTrainDataset` and `WSSValDataset`:
    - Supports single split (`splits/single`) or k-fold splits (`splits/{n_folds}/fold_{fold}`).
  - Configures:
    - A **multi-label balanced sampler** (`MultiLabelBalancedSampler`) or a standard `RandomSampler`.
    - Custom collates for:
      - Training: `(image, label_with_bg, bg_mask)`.
      - Validation: `(image, label_with_bg, seg_masks)`.
  - Exposes `train_dataloader()` and `val_dataloader()` with the configured batch size, workers, pinning, etc.

### `model.py`

- Defines the core LightningModule: **`WSSSegmentationModel`**.
- Architecture:
  - Two students (`model1`, `model2`), each a `SegformerMultiTask` instance.
  - Optionally share a backbone (controlled via config).
- Losses and training stages:
  - Stage 0 (early epochs):
    - Focus on classification and feature discrepancy loss between students.
  - Stage 2 (later epochs):
    - Generates hard pseudo-labels for segmentation using CAMs + TTA.
    - Uses a cross-teaching scheme: each student supervises the other.
  - Uses configurable weights (`lambda_cls`, `lambda_seg`, `lambda_dis`).
- Metrics:
  - Multi-label classification metrics (`MultilabelMetrics`) for both training and validation.
  - mIoU metrics (`MiouMetric`) for both segmentation outputs and CAMs.
- Optimization:
  - Builds parameter groups for backbone vs. heads (segmentation + classifier).
  - Uses `AdamW` with a cosine annealing LR scheduler.

### `backbone.py`

- Defines **`SegformerMultiTask`**:
  - Creates:
    - A SegFormer classification model (`SegformerForImageClassification`) and reuses its encoder as the backbone.
    - A SegFormer segmentation model (`SegformerForSemanticSegmentation`) and adapts its decode head to `num_classes`.
  - Uses `TransformerGradCAM` (`helpers`) to generate CAMs.
  - Forward pass returns:
    - Classification logits.
    - Segmentation logits.
    - CAMs and last feature map.

### `augmentation.py`

- **`cutmix_with_masks`**:
  - Performs CutMix between two mini-batches (`x1`, `x2`) and their multi-label annotations.
  - Produces:
    - Mixed images.
    - Combined multi-label targets.
    - Spatial CutMix mask.
    - Per-sample lists of foreground class indices coming from each original image.
- **`mixup_with_masks`**:
  - Performs MixUp between two mini-batches.
  - Produces:
    - Mixed images.
    - Mixed multi-label targets (thresholded).
    - Per-sample mixing coefficients and involved classes.

### `callbacks.py`

- **`MemoryLogCallback`**:
  - Logs:
    - RSS (RAM) from `resource.getrusage`.
    - VRAM usage from `torch.cuda.memory_allocated()`.
  - Writes CSV rows with epoch, global step, phase (`train`/`val`), batch index, RSS, and VRAM.
- **`SaveEveryEpochCallback`**:
  - Saves a checkpoint (`epoch_XX.ckpt`) at the end of every epoch in a configurable `dirpath`.

### `metrics.py`

- **`MiouMetric`**:
  - Manually computes:
    - IoU for each class.
    - Mean IoU over all classes.
  - Supports an `ignore_index` to skip pixels.
  - Optionally logs metrics via a PyTorch Lightning logger.

### `helpers/transforms.py`

- **`MaskAwareTransforms`** (training):
  - Applies image augmentations such as horizontal/vertical flips, random rotations, and random resized crops.
  - Propagates and updates a **valid-region mask** through all transformations.
  - Normalizes images with ImageNet-style mean/std and returns `(image_tensor, mask_tensor)`.
- **`DomainCropAugmentation`**:
  - Domain-specific augmentation for this fast-food scenario.
  - Divides the image into spatial regions (top/bottom/center, etc.).
  - Depending on present classes and probabilities:
    - Crops or obscures specific regions.
    - Zeroes out incompatible classes in the multi-hot label vector.
  - Returns modified image, label, and mask.
- **`MaskAwareValidationTransforms`**:
  - Simple resize + normalization for validation images.
  - Also maintains a valid-region mask.

### Other helpers

- `helpers/mask_loss_utils.py`: utilities to apply `ignore_index` to segmentation masks.
- `helpers/sampler.py`: implements `MultiLabelBalancedSampler` for class-balanced sampling.
- `helpers/metrics/classification_v2.py`: multi-label classification metrics (precision, recall, F1, mAP, etc.).
- `helpers/vit_grad_cam.py` and `helpers/cam_processing.py`: Grad-CAM generation and post-processing helpers used by `SegformerMultiTask` and the training loop.

---

## Quick command reference

- **Create environment with `uv`**:

```bash
uv sync
```

- **Run training on the mini sample dataset**:

```bash
uv run python train.py --config configs/mini.json --dataset-root datasets/WeakSupervisionSegmentation-mini:v1
```

- **Convert FoodSeg103 to the local format**:

```bash
uv run python scripts/convert_foodseg103.py --output-dir datasets/FoodSeg103_converted
```

- **Run training on the converted FoodSeg103 dataset**:

```bash
uv run python train.py --config configs/default.json --dataset-root datasets/FoodSeg103_converted
```

---

## Hardware used for experiments

The main experiments and end-to-end tests for this repository were run on a Linux workstation similar to the one hosting this repo, with the following characteristics:

- **Operating system**: Linux (kernel `6.14.0-37-generic`).
- **Python**: 3.10+ (matching the `>=3.10` requirement in `pyproject.toml`).
- **GPU**: Single NVIDIA GPU with CUDA support (tested with a 20–24 GB VRAM class card).
- **CPU**: Multi-core desktop-class CPU (≥ 8 physical cores).
- **RAM**: At least 32 GB system memory.

You can train smaller configurations (e.g. the mini dataset) on more modest hardware, but for full-resolution training on larger datasets like FoodSeg103 or the private VSS dataset, a CUDA-capable GPU with ≥ 20 GB VRAM and ample RAM is strongly recommended.

---

