# VSS Dataset Description

## Overview

The VSS (Visual Sandwich Assembly) dataset addresses **fast-food sandwich assembly** in an industrial setting. Images are acquired from **CCTV, top-down** cameras. The task is **ingredient-level semantic segmentation**. Annotations are provided as **segmentation masks** (pixel-level) for a validation subset and as **image-level labels** (multi-label classification) for the training subset. The full dataset cannot be released due to commercial restrictions.

## Dataset Size

- **Total images (with pixel-level annotations):** 2185 (full validation set).
- **Total images (training set, image-level only):** 11605 (per train fold; 4-fold cross-validation).
- **Number of classes:** 17 (including background).

Per-split statistics (training only) are provided in this folder as CSV and JSON files with the suffix `_split_0` through `_split_{split_n - 1}`.

## Image Properties

Mean resolution (validation set): **354 × 299** pixels. Images are captured under industrial conditions (fixed viewpoint, variable lighting, and possible occlusion from assembly).

## Class Distribution

Validation tables refer to the **full validation set**. Training tables refer to **split 0** (use the corresponding `dataset_statistics_train_split_*.csv` for other folds).

### Validation set (pixel-level annotations)

| Class | Images | Pixel % |
|-------|--------|--------|
| Pao | 2179 | 20.2212 |
| background | 2185 | 20.1802 |
| Alface | 1602 | 12.2021 |
| Queijo | 2006 | 11.6597 |
| Tomate | 1290 | 6.3546 |
| Carne | 1724 | 6.3381 |
| Mao_Pessoa | 725 | 5.0195 |
| Molho | 632 | 3.7786 |
| Cebola | 829 | 3.2806 |
| Pepino | 620 | 2.8079 |
| Cheddar | 942 | 2.3652 |
| Cream-cheese | 406 | 1.8663 |
| Processados | 175 | 1.2382 |
| Pimentao | 421 | 1.1643 |
| Azeitona | 529 | 0.8155 |
| Bacon | 97 | 0.451 |
| Picles | 96 | 0.2568 |

### Training set (image-level annotations)

Pixel percentage is not defined for image-level annotations.

| Class | Images |
|-------|--------|
| background | 11605 |
| Picles | 400 |
| Pao | 11385 |
| Cebola | 3850 |
| Queijo | 10644 |
| Carne | 8641 |
| Cheddar | 4486 |
| Azeitona | 2463 |
| Pimentao | 1759 |
| Molho | 2939 |
| Alface | 7593 |
| Mao_Pessoa | 3951 |
| Cream-cheese | 2232 |
| Tomate | 5774 |
| Processados | 965 |
| Bacon | 574 |
| Pepino | 2274 |

## Dataset Characteristics

- **Validation set:** average classes per image = 7.53; average mask coverage = 100.00%.
- **Training set:** average classes per image = 7.03.
- The dataset exhibits class imbalance, small or low-contrast ingredients, and overlapping regions (multiple ingredients in the same scene).

## Annotation Protocol

Images were annotated using **Label Studio**. The annotations were exported and **converted to YOLO format** (text files with polygon coordinates per object, one line per instance). At runtime, these YOLO files are rasterized to per-class binary masks. For faster loading, rasterized masks are **cached as a single PNG per image** in the same directory as the source `.txt` file; each pixel in the PNG stores the class index (0 = background, 1–16 = ingredients). When the cache exists, masks are read from the PNG instead of re-parsing the YOLO files.

## Public Subset

A representative subset of up to 30 images is provided to illustrate dataset structure and annotations.

## Reproducibility Note

While the full dataset is not publicly available, we provide code, configuration files, dataset statistics, and representative samples to enable reproducibility of the methodology.
