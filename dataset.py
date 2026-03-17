"""
Datasets for WSSS in the local format (dataset.yaml + splits + YOLO labels).
See plans/dataset.md for the format specification.
"""

import json
import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

try:
    import yaml
except ImportError:
    yaml = None

try:
    import cv2
except ImportError:
    cv2 = None

from helpers.transforms import MaskAwareTransforms, MaskAwareValidationTransforms, DomainCropAugmentation


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------

class DatasetConfig:
    """
    Reads and validates dataset.yaml; exposes num_classes, num_foreground_classes,
    class_names and dataset paths.
    """
    def __init__(self, dataset_root):
        self.dataset_root = os.path.abspath(dataset_root)
        self.yaml_path = os.path.join(self.dataset_root, "dataset.yaml")
        if not os.path.isfile(self.yaml_path):
            raise FileNotFoundError(f"dataset.yaml not found: {self.yaml_path}")
        if yaml is None:
            raise ImportError("PyYAML is required to read dataset.yaml. Install with: pip install pyyaml")
        with open(self.yaml_path) as f:
            meta = yaml.safe_load(f)
        self._meta = meta
        self.num_classes = int(meta["num_classes"])
        self.num_foreground_classes = int(meta.get("num_foreground_classes", self.num_classes - 1))
        names = meta.get("class_names", {})
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}
        else:
            self.class_names = {i: str(names[i]) for i in range(len(names))} if names else {}
        self.image_format = meta.get("image_format", "jpg")
        self.train_images_dir = os.path.join(self.dataset_root, "train", "images")
        self.val_images_dir = os.path.join(self.dataset_root, "val", "images")
        self.val_labels_dir = os.path.join(self.dataset_root, "val", "labels")
        self.annotations_dir = os.path.join(self.dataset_root, "annotations")
        self.splits_dir = os.path.join(self.dataset_root, "splits")

    def get_single_train_split_file(self):
        return os.path.join(self.splits_dir, "single", "train.txt")

    def get_single_train_labels_file(self):
        return os.path.join(self.annotations_dir, "train_labels.json")

    def get_kfold_train_split_file(self, n_folds, fold):
        return os.path.join(self.splits_dir, str(n_folds), f"fold_{fold}", "train.txt")

    def get_kfold_train_labels_file(self, n_folds, fold):
        return os.path.join(self.splits_dir, str(n_folds), f"fold_{fold}", "train_labels.json")


# ---------------------------------------------------------------------------
# YOLO masks: load and cache (largest area first; atomic write)
# ---------------------------------------------------------------------------

def load_yolo_masks(txt_path, num_classes, width, height):
    """
    Loads YOLO polygons from the .txt file and generates per-class masks.
    Polygons are drawn from LARGEST area to SMALLEST so that smaller classes
    remain visible over larger ones (as specified in dataset.md).
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for load_yolo_masks. Install with: pip install opencv-python")
    masks = np.zeros((num_classes, height, width), dtype=np.uint8)
    polygons_by_class = {}  # class_id -> list of (coords_xy, area)
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            if class_id < 0 or class_id >= num_classes:
                continue
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
            coords_px = (coords * [width, height]).astype(np.int32)
            area = cv2.contourArea(coords_px)
            if area <= 0:
                continue
            if class_id not in polygons_by_class:
                polygons_by_class[class_id] = []
            polygons_by_class[class_id].append((coords_px, area))
    # Ordenar por área decrescente (maior primeiro) e desenhar
    for class_id, items in polygons_by_class.items():
        items_sorted = sorted(items, key=lambda x: -x[1])
        for coords_px, _ in items_sorted:
            cv2.fillPoly(masks[class_id], [coords_px], 1)
    return masks


def generate_mask_cache(txt_path, png_path, num_classes, width, height):
    """
    Generates a .png cache of the mask from the .txt file.
    Uses atomic write (tempfile + rename) to avoid corruption with multiple workers.
    If png_path already exists and is newer than txt_path, it is not regenerated.
    """
    if os.path.isfile(png_path):
        try:
            if os.path.getmtime(png_path) >= os.path.getmtime(txt_path):
                return
        except OSError:
            pass
    masks = load_yolo_masks(txt_path, num_classes, width, height)
    # Save (H, W) with pixel value = class_id (last drawn class wins; we already draw larger->smaller)
    out = np.zeros((height, width), dtype=np.uint16)
    for c in range(num_classes):
        out[masks[c] > 0] = c
    dirname = os.path.dirname(png_path)
    os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=dirname)
    try:
        with os.fdopen(fd, "wb") as f:
            Image.fromarray(out).save(f, format="PNG")
        os.replace(tmp, png_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_mask_from_cache_or_txt(txt_path, png_path, num_classes, width, height):
    """
    Loads masks: uses .png cache if it exists and is valid; otherwise generates and saves it.
    Returns array (num_classes, H, W) float32 in [0, 1].
    """
    generate_mask_cache(txt_path, png_path, num_classes, width, height)
    if not os.path.isfile(png_path):
        masks = load_yolo_masks(txt_path, num_classes, width, height)
    else:
        img = np.array(Image.open(png_path))
        if img.ndim == 3:
            img = img[:, :, 0]
        img = img.astype(np.int32)
        h, w = img.shape[0], img.shape[1]
        masks = np.zeros((num_classes, h, w), dtype=np.float32)
        for c in range(num_classes):
            masks[c] = (img == c).astype(np.float32)
        if (w, h) != (width, height):
            masks_resized = np.zeros((num_classes, height, width), dtype=np.float32)
            for c in range(num_classes):
                m = Image.fromarray((masks[c] * 255).astype(np.uint8))
                m = m.resize((width, height), Image.NEAREST)
                masks_resized[c] = np.array(m) / 255.0
            masks = masks_resized
    return masks.astype(np.float32)


# ---------------------------------------------------------------------------
# WSSTrainDataset
# ---------------------------------------------------------------------------

class WSSTrainDataset(Dataset):
    """
    Training dataset built from split .txt and .json labels (local format).
    __getitem__ returns (image, label_multihot_with_bg, bg_mask).
    JSON has num_foreground_classes dims (without bg); the loader adds bg at index 0.
    """
    def __init__(self, dataset_root, split_file, labels_file, dataset_config, transform=None, image_size=448, domain_crop_config=None):
        self.dataset_root = dataset_root
        self.config = dataset_config
        self.num_classes = dataset_config.num_classes
        self.num_foreground = dataset_config.num_foreground_classes
        self.image_size = image_size
        self.image_format = dataset_config.image_format
        self.train_images_dir = dataset_config.train_images_dir
        with open(split_file) as f:
            self.image_names = [line.strip() for line in f if line.strip()]
        with open(labels_file) as f:
            self.labels_dict = json.load(f)
        self.transform = transform
        if self.transform is None:
            self.transform = MaskAwareTransforms(image_size=image_size)
        # Domain crop augmentation (enabled via config)
        self.domain_crop = None
        if domain_crop_config and domain_crop_config.get("enabled", False):
            self.domain_crop = DomainCropAugmentation(domain_crop_config, image_size=image_size)

    def __len__(self):
        return len(self.image_names)

    def _get_image_path(self, name):
        ext = self.image_format if self.image_format else "jpg"
        return os.path.join(self.train_images_dir, f"{name}.{ext}")

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = self._get_image_path(name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        # Label: JSON has num_foreground_classes positions (without bg)
        vec = self.labels_dict.get(name, [0] * self.num_foreground)
        if len(vec) != self.num_foreground:
            vec = (vec + [0] * self.num_foreground)[:self.num_foreground]
        label_fg = np.array(vec, dtype=np.float32)
        w, h = image.size
        mask = np.ones((h, w), dtype=np.float32)
        # Domain crop: modifies image, label_fg and mask before label_with_bg
        if self.domain_crop:
            image, label_fg, mask = self.domain_crop(image, label_fg, mask)
        # Add background at index 0 (always 1.0)
        label_with_bg = np.zeros(self.num_classes, dtype=np.float32)
        label_with_bg[0] = 1.0
        label_with_bg[1 : 1 + self.num_foreground] = label_fg
        if self.transform:
            image, mask = self.transform(image, mask)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        label_with_bg = torch.from_numpy(label_with_bg).float()
        return image, label_with_bg, mask

    def get_labels(self):
        """Returns a list of labels (with bg) for the sampler."""
        labels = []
        for name in self.image_names:
            vec = self.labels_dict.get(name, [0] * self.num_foreground)
            if len(vec) != self.num_foreground:
                vec = (vec + [0] * self.num_foreground)[:self.num_foreground]
            label_fg = np.array(vec, dtype=np.float32)
            label_with_bg = np.zeros(self.num_classes, dtype=np.float32)
            label_with_bg[0] = 1.0
            label_with_bg[1 : 1 + self.num_foreground] = label_fg
            labels.append(label_with_bg)
        return np.array(labels)


# ---------------------------------------------------------------------------
# WSSValDataset
# ---------------------------------------------------------------------------

class WSSValDataset(Dataset):
    """
    Validation dataset: val/images/ + val/labels/ (YOLO .txt with .png cache).
    __getitem__ returns (image, label_multihot_with_bg, masks[num_classes, H, W], idx).
    """
    def __init__(self, dataset_root, dataset_config, transform=None, image_size=448):
        self.dataset_root = dataset_root
        self.config = dataset_config
        self.num_classes = dataset_config.num_classes
        self.num_foreground = dataset_config.num_foreground_classes
        self.image_size = image_size
        self.image_format = dataset_config.image_format
        self.val_images_dir = dataset_config.val_images_dir
        self.val_labels_dir = dataset_config.val_labels_dir
        if not os.path.isdir(self.val_images_dir):
            raise FileNotFoundError(f"Validation images directory not found: {self.val_images_dir}")
        self.image_names = []
        for f in os.listdir(self.val_images_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                base = f
                for ext in (".jpg", ".jpeg", ".png"):
                    if base.lower().endswith(ext):
                        base = base[: -len(ext)]
                        break
                self.image_names.append(base)
        self.image_names = sorted(set(self.image_names))
        self.transform = transform
        if self.transform is None:
            self.transform = MaskAwareValidationTransforms(image_size=image_size)

    def __len__(self):
        return len(self.image_names)

    def _get_image_path(self, name):
        for ext in (self.image_format, "jpg", "jpeg", "png"):
            p = os.path.join(self.val_images_dir, f"{name}.{ext}")
            if os.path.isfile(p):
                return p
        return os.path.join(self.val_images_dir, f"{name}.{self.image_format or 'jpg'}")

    def _get_txt_path(self, name):
        return os.path.join(self.val_labels_dir, f"{name}.txt")

    def _get_cache_png_path(self, name):
        return os.path.join(self.val_labels_dir, f"{name}.png")

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = self._get_image_path(name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Validation image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        txt_path = self._get_txt_path(name)
        png_path = self._get_cache_png_path(name)
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"Validation label not found: {txt_path}")
        masks = load_mask_from_cache_or_txt(
            txt_path, png_path, self.num_classes, self.image_size, self.image_size
        )
        # Multihot label from masks (present classes)
        label_with_bg = np.zeros(self.num_classes, dtype=np.float32)
        label_with_bg[0] = 1.0
        for c in range(1, self.num_classes):
            if masks[c].sum() > 0:
                label_with_bg[c] = 1.0
        if self.transform:
            image, _ = self.transform(image, None)
        masks = torch.from_numpy(masks).float()
        if masks.shape[1:] != (self.image_size, self.image_size):
            masks = masks.unsqueeze(0)
            masks = F.interpolate(masks, size=(self.image_size, self.image_size), mode="nearest")
            masks = masks.squeeze(0)
        label_with_bg = torch.from_numpy(label_with_bg).float()
        return image, label_with_bg, masks, idx


# ---------------------------------------------------------------------------
# Compatibility: keep old names for existing imports
# ---------------------------------------------------------------------------

# For datamodule and train that may still reference old names:
# WSSSegmentationDataset -> local (new) use is WSSTrainDataset
# WSSSegmentationMaskDataset -> WSSValDataset
# When datamodule is rewritten, use only WSSTrainDataset and WSSValDataset.

__all__ = [
    "DatasetConfig",
    "load_yolo_masks",
    "generate_mask_cache",
    "WSSTrainDataset",
    "WSSValDataset",
]
