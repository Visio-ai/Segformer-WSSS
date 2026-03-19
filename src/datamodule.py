"""
WSSS DataModule for local format (dataset.yaml + splits + YOLO).
Supports single mode (splits/single/) and k-fold mode (splits/{n_folds}/fold_{fold}/).
"""

import os
import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler

from src.dataset import DatasetConfig, WSSTrainDataset, WSSValDataset
from src.helpers.transforms import MaskAwareTransforms, MaskAwareValidationTransforms
from src.helpers.sampler import MultiLabelBalancedSampler


class WSSDataModule(pl.LightningDataModule):
    """
    DataModule for WSSS with local dataset.
    - __init__ receives dataset_root, config, and optionally fold, n_folds.
    - Single mode: splits/single/train.txt + annotations/train_labels.json
    - K-fold mode: splits/{n_folds}/fold_{fold}/train.txt + train_labels.json
    - Validation always uses: val/images/ + val/labels/
    """

    def __init__(
        self,
        dataset_root,
        config,
        batch_size=8,
        image_size=448,
        use_balanced_sampler=True,
        sampler_num_samples=3000,
        fold=None,
        n_folds=None,
        num_workers=4,
        persistent_workers=True,
    ):
        super().__init__()
        self.dataset_root = os.path.abspath(dataset_root)
        self.config = config
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_balanced_sampler = use_balanced_sampler
        self.sampler_num_samples = sampler_num_samples
        self.fold = fold
        self.n_folds = n_folds
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.current_epoch = 0
        self._dataset_config = None
        self.train_dataset = None
        self.val_dataset = None

    def _get_dataset_config(self):
        if self._dataset_config is None:
            self._dataset_config = DatasetConfig(self.dataset_root)
        return self._dataset_config

    def setup(self, stage=None):
        dsc = self._get_dataset_config()
        image_size = self.config.get("image_size", self.image_size)
        transform_train = MaskAwareTransforms(image_size=image_size)
        transform_val = MaskAwareValidationTransforms(image_size=image_size)

        if self.n_folds is not None and self.fold is not None:
            split_file = dsc.get_kfold_train_split_file(self.n_folds, self.fold)
            labels_file = dsc.get_kfold_train_labels_file(self.n_folds, self.fold)
        else:
            split_file = dsc.get_single_train_split_file()
            labels_file = dsc.get_single_train_labels_file()

        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split de treino não encontrado: {split_file}")
        if not os.path.isfile(labels_file):
            raise FileNotFoundError(f"Labels de treino não encontrados: {labels_file}")

        domain_crop_config = self.config.get("domain_crop", None)
        self.train_dataset = WSSTrainDataset(
            self.dataset_root,
            split_file=split_file,
            labels_file=labels_file,
            dataset_config=dsc,
            transform=transform_train,
            image_size=image_size,
            domain_crop_config=domain_crop_config,
        )
        self.val_dataset = WSSValDataset(
            self.dataset_root,
            dataset_config=dsc,
            transform=transform_val,
            image_size=image_size,
        )

    def cls_collate_fn(self, batch):
        """Collate for training: (image, label_multihot_with_bg, bg_mask)."""
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        masks = [item[2] for item in batch]
        if isinstance(labels[0], torch.Tensor):
            labels_t = torch.stack(labels, dim=0)
        else:
            labels_t = torch.tensor(np.array(labels, dtype=np.float32))
        return (
            torch.stack(images, dim=0),
            labels_t,
            torch.stack(masks, dim=0),
        )

    def mask_collate_fn(self, batch):
        """Collate for validation: (image, label_multihot_with_bg, masks)."""
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        seg_masks = [item[2] for item in batch]
        if isinstance(labels[0], torch.Tensor):
            labels_t = torch.stack(labels, dim=0)
        else:
            labels_t = torch.tensor(np.array(labels, dtype=np.float32))
        return (
            torch.stack(images, dim=0),
            labels_t,
            torch.stack(seg_masks, dim=0),
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        if self.use_balanced_sampler:
            labels = self.train_dataset.get_labels()
            sampler = MultiLabelBalancedSampler(
                labels,
                max_samples=self.sampler_num_samples,
                num_classes=self.train_dataset.num_classes,
                epoch=self.current_epoch,
            )
            return DataLoader(
                self.train_dataset,
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.cls_collate_fn,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            )
        sampler = RandomSampler(
            self.train_dataset,
            num_samples=self.sampler_num_samples,
        )
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.cls_collate_fn,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.mask_collate_fn,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
