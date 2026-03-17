"""
WSS LightningModule (Weakly Supervised Semantic Segmentation).
"""

import math
import os
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from augmentation import cutmix_with_masks
from backbone import SegformerMultiTask
from helpers.mask_loss_utils import apply_ignore_index_to_segmentation
from helpers.metrics.classification_v2 import MultilabelMetrics
from last_losses import discrepancy_loss
from metrics import MiouMetric
from tta import generate_hard_pseudo_labels_tta as generate_hard_pseudo_labels_tta_fn

ENABLE_TIMING_PRINTS = os.getenv("ENABLE_TIMING_PRINTS", "false").lower() == "true"


class WSSSegmentationModel(pl.LightningModule):
    """LightningModule for WSSS training."""

    def __init__(self, num_classes, lr=2e-5, image_size=448, freeze_backbone=False, config=None):
        super().__init__()
        self.config = config or {}
        cfg = self.config
        backbone_name = cfg.get("backbone_model", "nvidia/mit-b0")
        shared_backbone = cfg.get("shared_backbone", True)
        self.model1 = SegformerMultiTask(num_classes, model_name=backbone_name, image_size=image_size, shared_backbone=shared_backbone)
        self.model2 = SegformerMultiTask(num_classes, model_name=backbone_name, image_size=image_size, shared_backbone=shared_backbone)
        self.model = self.model1
        self.lr = lr
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.model1.backbone.parameters():
                param.requires_grad = False
            for param in self.model2.backbone.parameters():
                param.requires_grad = False
            print("Backbone congelado")

        self.cls_loss_fn = lambda pred, target: F.multilabel_soft_margin_loss(pred, target, reduction="mean")
        foreground_weight = cfg.get("foreground_weight", 1.0)
        class_weights = torch.ones(num_classes) * foreground_weight
        class_weights[0] = cfg.get("background_weight", 1.0)
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
        self.bg_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        self.lambda_dis = cfg.get("lambda_dis", 0.1)
        self.tau_l = cfg.get("tau_l", 0.2)
        self.tauh0 = cfg.get("tauh0", 0.7)
        self.tauhT = cfg.get("tauhT", 0.55)
        self.count = np.zeros(num_classes - 1)
        self.count_with_bg = np.zeros(num_classes)
        self.validation_outputs = []
        class_names_cls = [f"class_{i}" for i in range(1, num_classes)]
        self.val_classification_metrics = MultilabelMetrics(class_names_cls, logger=self, threshold=0.5)
        self.training_metrics = MultilabelMetrics(class_names_cls, logger=self, threshold=0.5)
        self.val_cam_metrics = MiouMetric(num_classes=num_classes, ignore_index=255, logger=self)
        self.miou_metric = MiouMetric(num_classes=num_classes, ignore_index=255, logger=self)
        self.save_hyperparameters()

    def forward(self, x, target_labels=None, test=False):
        return self.model(x, target_labels, test=test)

    def tauh_cosine(self, global_step, total_steps):
        tauh0 = self.tauh0
        tauhT = self.tauhT
        t = float(global_step)
        T = float(max(1, total_steps))
        return tauh0 - 0.5 * (tauh0 - tauhT) * (1 - math.cos(math.pi * t / T))

    def on_train_start(self):
        """Computes and caches total_steps once at the beginning of training."""
        super().on_train_start()
        try:
            self._cached_total_steps = float(
                self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
            )
        except Exception:
            self._cached_total_steps = 20000.0
        print(f"Total cached steps: {self._cached_total_steps}")

    def on_train_epoch_start(self):
        current_epoch = self.trainer.current_epoch
        if hasattr(self.trainer.datamodule, "current_epoch"):
            self.trainer.datamodule.current_epoch = current_epoch

        self.images_without_bg_count = 0
        self.seg_bg_stats = []
        self.gt_bg_stats = []
        self.count = np.zeros(self.num_classes - 1)
        self.validation_outputs = []
        self.miou_metric.reset()
        self.val_cam_metrics.reset()

        for i, group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"lr_group_{i}", group["lr"], prog_bar=True)

        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        step_start = time.time()

        if batch_idx == 0:
            self.count = np.zeros(self.num_classes - 1)
            self.count_with_bg = np.zeros(self.num_classes)
            try:
                self.log("cur_epoch", float(self.current_epoch), prog_bar=True)
            except Exception:
                pass

        stage_transition_pct = self.config.get("stage_transition_pct", 0.3)
        progress = float(self.current_epoch) / max(1.0, float(self.trainer.max_epochs))
        if progress < stage_transition_pct:
            stage = 0
        else:
            stage = 2

        data_time = time.time() - step_start

        if stage == 0:
            return self._training_stage0(batch, batch_idx, data_time, step_start)
        if len(batch) >= 3 and stage != 0:
            return self._training_stage2(batch, batch_idx, data_time, step_start)
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def _training_stage0(self, batch, batch_idx, data_time, step_start):
        x, y_cls, bg_mask = batch[:3]
        y_cls_with_bg = y_cls

        aug_start = time.time()
        cutmix_prob_s0 = self.config.get("cutmix", {}).get("probability_stage0", 0.1)
        if random.random() < cutmix_prob_s0:
            indices = torch.randperm(x.shape[0], device=x.device)
            x2 = x[indices]
            y2 = y_cls_with_bg[indices]
            x, y_cls_with_bg, mask_spatial, y1_classes, y2_classes = cutmix_with_masks(
                x, y_cls_with_bg, x2, y2, alpha=1.0
            )
            y_cls_with_bg = (y_cls_with_bg > 0.2).float()
        aug_time = time.time() - aug_start

        y_cls = y_cls_with_bg
        images_without_bg = (y_cls[:, 0] < 0.3).sum().item()
        self.images_without_bg_count += images_without_bg
        y_cls_fg = y_cls[:, 1:]
        self.count_with_bg += (y_cls > 0.5).sum(dim=0).cpu().numpy()
        self.count += y_cls_fg.sum(dim=0).cpu().numpy()

        forward_start = time.time()
        out1 = self.model1(x, target_labels=[], test=False)
        out2 = self.model2(x, target_labels=[], test=False)
        forward_time = time.time() - forward_start

        metrics_start = time.time()
        self.training_metrics.update(preds=out1["classification"][:, 1:], targets=y_cls_fg)
        self.training_metrics.update(preds=out2["classification"][:, 1:], targets=y_cls_fg)
        metrics_time = time.time() - metrics_start

        loss_start = time.time()
        loss_cls_1 = self.cls_loss_fn(out1["classification"][:, 1:], y_cls_fg)
        loss_cls_2 = self.cls_loss_fn(out2["classification"][:, 1:], y_cls_fg)
        loss_cls = loss_cls_1 + loss_cls_2
        f1 = out1["features_map"]
        f2 = out2["features_map"]
        loss_dis = discrepancy_loss(f1, f2)
        lambda_cls = self.config.get("lambda_cls", 20.0)
        total_loss = lambda_cls * loss_cls + loss_dis
        loss_time = time.time() - loss_start
        total_time = time.time() - step_start

        if batch_idx == 0:
            self.log("train/timing/data", data_time)
            self.log("train/timing/aug", aug_time)
            self.log("train/timing/forward", forward_time)
            self.log("train/timing/metrics", metrics_time)
            self.log("train/timing/loss", loss_time)
            self.log("train/timing/total", total_time)

        if ENABLE_TIMING_PRINTS:
            print(f"[TIMING] Train Batch {batch_idx} - Data loading: {data_time:.4f}s")
            print(f"[TIMING] Train Batch {batch_idx} - Augmentation: {aug_time:.4f}s")
            print(f"[TIMING] Train Batch {batch_idx} - Forward pass: {forward_time:.4f}s")
            print(f"[TIMING] Train Batch {batch_idx} - Metrics: {metrics_time:.4f}s")
            print(f"[TIMING] Train Batch {batch_idx} - Loss calculation: {loss_time:.4f}s")
            print(f"[TIMING] Train Batch {batch_idx} - Total step: {total_time:.4f}s")

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_loss_cls", loss_cls)
        self.log("train_loss_dis", loss_dis)
        self.log("train_epoch", float(self.current_epoch))
        self.log("train_global_step", float(self.global_step))
        return total_loss

    def _training_stage2(self, batch, batch_idx, data_time, step_start):
        # These are not gt masks, but masks that show where the fake background from augmentation is
        x, y_cls, masks = batch[:3]
        y_cls_with_bg = y_cls
        y_cls_fg = y_cls[:, 1:]
        aug_info = None
        aug_start = time.time()
        cutmix_prob_s2 = self.config.get("cutmix", {}).get("probability_stage2", 0.5)
        if random.random() < cutmix_prob_s2:
            indices = torch.randperm(x.shape[0], device=x.device)
            x2 = x[indices]
            y2 = y_cls_with_bg[indices]
            masks2 = masks[indices] if masks is not None else None
            x, y_cls_with_bg, mask_spatial, y1_classes, y2_classes = cutmix_with_masks(
                x, y_cls_with_bg, x2, y2, alpha=1.0
            )
            if masks is not None and masks2 is not None:
                if mask_spatial.shape[-2:] != masks.shape[-2:]:
                    mask_spatial_resized = F.interpolate(
                        mask_spatial.unsqueeze(1).float(),
                        size=masks.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                else:
                    mask_spatial_resized = mask_spatial.float()
                masks = torch.where(
                    mask_spatial_resized >= 0.5,
                    masks,
                    masks2,
                )
            aug_info = {
                "type": "cutmix",
                "mask_spatial": mask_spatial,
                "y1_classes": y1_classes,
                "y2_classes": y2_classes,
            }
        y_cls = y_cls_with_bg
        y_cls_fg = y_cls[:, 1:]
        aug_time = time.time() - aug_start

        self.images_without_bg_count += (y_cls[:, 0] < 0.3).sum().item()
        self.count += y_cls_fg.sum(dim=0).cpu().numpy()
        self.count_with_bg += (y_cls > 0.5).sum(dim=0).cpu().numpy()

        gen_cams = y_cls
        forward_start = time.time()
        out1 = self.model1(x, target_labels=gen_cams, test=False)
        out2 = self.model2(x, target_labels=gen_cams, test=False)
        forward_time = time.time() - forward_start

        metrics_start = time.time()
        self.training_metrics.update(preds=out1["classification"][:, 1:], targets=y_cls_fg)
        self.training_metrics.update(preds=out2["classification"][:, 1:], targets=y_cls_fg)
        metrics_time = time.time() - metrics_start

        loss_start = time.time()
        loss_cls_1 = self.cls_loss_fn(out1["classification"][:, 1:], y_cls_fg)
        loss_cls_2 = self.cls_loss_fn(out2["classification"][:, 1:], y_cls_fg)
        loss_cls = loss_cls_1 + loss_cls_2
        f1 = out1["features_map"]
        f2 = out2["features_map"]
        loss_dis = discrepancy_loss(f1, f2)

        total_steps = getattr(self, "_cached_total_steps", 20000.0)
        stage_transition_pct = self.config.get("stage_transition_pct", 0.3)
        step_limit = total_steps * stage_transition_pct
        tauh_current = self.tauh_cosine(self.global_step - step_limit, total_steps - step_limit)
        self.log("tauh_current", tauh_current)

        ignore_index = 255
        ce_loss = self.seg_loss_fn
        cams1_ups = F.interpolate(out1["cams"], size=out1["segmentation"].shape[2:], mode="bilinear", align_corners=False)
        cams2_ups = F.interpolate(out2["cams"], size=out2["segmentation"].shape[2:], mode="bilinear", align_corners=False)

        if aug_info is not None:
            cams1_ups = cams1_ups.clone()
            cams2_ups = cams2_ups.clone()
            if aug_info["type"] == "cutmix":
                mask_spatial = aug_info["mask_spatial"]
                y1_classes = aug_info["y1_classes"]
                y2_classes = aug_info["y2_classes"]
                H_cam, W_cam = cams1_ups.shape[2], cams1_ups.shape[3]
                mask_spatial_resized = F.interpolate(
                    mask_spatial.unsqueeze(1).float(),
                    size=(H_cam, W_cam),
                    mode="nearest",
                ).squeeze(1)
                for b in range(x.shape[0]):
                    for cls_idx in y1_classes[b]:
                        model_cls_idx = cls_idx + 1
                        cams1_ups[b, model_cls_idx] = cams1_ups[b, model_cls_idx] * mask_spatial_resized[b]
                        cams2_ups[b, model_cls_idx] = cams2_ups[b, model_cls_idx] * mask_spatial_resized[b]
                    for cls_idx in y2_classes[b]:
                        model_cls_idx = cls_idx + 1
                        cams1_ups[b, model_cls_idx] = cams1_ups[b, model_cls_idx] * (1.0 - mask_spatial_resized[b])
                        cams2_ups[b, model_cls_idx] = cams2_ups[b, model_cls_idx] * (1.0 - mask_spatial_resized[b])

        tta_weights = self.config.get("tta", {}).get("weights")
        pseudo_labels1 = generate_hard_pseudo_labels_tta_fn(
            cams1_ups, y_cls, tauh_current, self.tau_l, ignore_index, self.model1, x,
            tta_weights=tta_weights, detach_cams=True,
        )
        del cams1_ups
        torch.cuda.empty_cache()
        pseudo_labels2 = generate_hard_pseudo_labels_tta_fn(
            cams2_ups, y_cls, tauh_current, self.tau_l, ignore_index, self.model2, x,
            tta_weights=tta_weights, detach_cams=True,
        )
        del cams2_ups
        torch.cuda.empty_cache()

        if masks is not None:
            masks_resized = F.interpolate(
                masks.unsqueeze(1),
                size=pseudo_labels2.shape[-2:],
                mode="nearest",
            ).squeeze(1)
            pseudo_labels2 = apply_ignore_index_to_segmentation(pseudo_labels2, masks_resized, 0)
            pseudo_labels1 = apply_ignore_index_to_segmentation(pseudo_labels1, masks_resized, 0)

        loss_seg_1 = ce_loss(out1["segmentation"], pseudo_labels2.detach())
        loss_seg_2 = ce_loss(out2["segmentation"], pseudo_labels1.detach())
        loss_seg_cross = loss_seg_1 + loss_seg_2

        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        lambda_cls = self.config.get("lambda_cls", 20.0)
        lambda_seg = self.config.get("lambda_seg", 0.3)
        total_loss = total_loss + lambda_cls * loss_cls
        total_loss = total_loss + lambda_seg * loss_seg_cross
        loss_time = time.time() - loss_start
        total_time = time.time() - step_start

        if ENABLE_TIMING_PRINTS:
            print(f"[TIMING] Train Batch {batch_idx} (Stage 2) - Data: {data_time:.4f}s, Forward: {forward_time:.4f}s, Loss: {loss_time:.4f}s, Total: {total_time:.4f}s")

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_loss_cls", loss_cls)
        self.log("train_loss_dis", loss_dis)
        self.log("train_loss_seg_cross", loss_seg_cross)
        return total_loss

    def on_train_epoch_end(self):
        self.training_metrics.log_metrics(self.global_step, prefix="train/")
        if hasattr(self, "images_without_bg_count"):
            self.log("train/images_without_bg", float(self.images_without_bg_count), prog_bar=False)
            print(f"\nEpoch {self.current_epoch} statistics:")
            print(f"   Images WITHOUT background: {self.images_without_bg_count}")
            if hasattr(self, "count_with_bg"):
                first10_counts = self.count_with_bg[:10].astype(int).tolist()
                print(f"   Samples per class (0-9, including BG): {first10_counts}")
            if hasattr(self, "count"):
                first10_fg_counts = self.count[:10].astype(int).tolist()
                print(f"   Samples per foreground class (0-9, without BG): {first10_fg_counts}")

    def _validation_miou_update(self, outputs, y_mask, target_size):
        """Updates mIoU metric (argmax of segmentation vs argmax of GT)."""
        if y_mask is None:
            return
        preds_seg = torch.sigmoid(
            F.interpolate(
                outputs["segmentation"],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        pred_indices = preds_seg.argmax(dim=1)
        target_indices = y_mask.argmax(dim=1)
        self.miou_metric.update(pred_indices, target_indices)

    def _validation_cam_metrics_update(self, outputs, y_mask, target_size):
        """Updates CAM mIoU metric (argmax of CAMs vs argmax of GT)."""
        if y_mask is None or outputs.get("cams") is None:
            return
        cams = outputs["cams"]
        cams_upsampled = torch.sigmoid(
            F.interpolate(cams, size=target_size, mode="bilinear", align_corners=False)
        ).detach()
        pred_indices = cams_upsampled.argmax(dim=1)
        target_indices = y_mask.argmax(dim=1)
        self.val_cam_metrics.update(pred_indices, target_indices)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_step_start = time.time()
        need_cams = batch_idx < 2
        # Grad-CAM precisa de autograd.grad; inference_mode só quando não calculamos CAMs (reduz pico de memória)
        inference_ctx = torch.inference_mode() if not need_cams else torch.enable_grad()
        with inference_ctx:
            if batch_idx == 0:
                print(f"[VAL] Starting validation epoch {self.current_epoch}")
                try:
                    self.log("cur_epoch", float(self.current_epoch), prog_bar=True)
                except Exception:
                    pass

            data_start = time.time()
            x, y_cls, y_mask = batch[:3]
            dev = next(self.parameters()).device
            x = x.to(dev)
            y_cls = y_cls.to(dev)
            y_mask = y_mask.to(dev) if y_mask is not None else None
            data_time = time.time() - data_start

            stage_transition_pct = self.config.get("stage_transition_pct", 0.3)
            progress = float(self.current_epoch) / max(1.0, float(self.trainer.max_epochs))
            stage = 0 if progress <= stage_transition_pct else 2

            forward_start = time.time()
            self.model = self.model.to(dev)
            target_labels_for_forward = [] if not need_cams else y_cls
            outputs = self.model(x, target_labels=target_labels_for_forward, test=False)
            forward_time = time.time() - forward_start

            loss_start = time.time()
            y_cls_fg = y_cls[:, 1:]
            loss_cls = self.cls_loss_fn(outputs["classification"][:, 1:], y_cls_fg)
            loss_time = time.time() - loss_start

            metrics_start = time.time()
            self.log("val_loss", loss_cls.detach(), prog_bar=True)
            self.val_classification_metrics.update(
                preds=outputs["classification"][:, 1:].detach(),
                targets=y_cls_fg,
            )
            metrics_time = time.time() - metrics_start

            seg_time = 0.0
            if stage == 2 and y_mask is not None:
                seg_start = time.time()
                target_size = (y_mask.shape[2], y_mask.shape[3]) if len(y_mask.shape) == 4 else (448, 448)
                self._validation_miou_update(outputs, y_mask, target_size)
                if need_cams:
                    self._validation_cam_metrics_update(outputs, y_mask, target_size)
                seg_time = time.time() - seg_start

            del outputs
            torch.cuda.empty_cache()

            total_time = time.time() - val_step_start
            if ENABLE_TIMING_PRINTS:
                print(f"[TIMING] Val Batch {batch_idx} - data={data_time:.4f}s forward={forward_time:.4f}s loss={loss_time:.4f}s total={total_time:.4f}s")
            if batch_idx == 0:
                self.log("val/timing/data", data_time)
                self.log("val/timing/forward", forward_time)
                self.log("val/timing/loss", loss_time)
                self.log("val/timing/metrics", metrics_time)
                self.log("val/timing/seg", seg_time)
                self.log("val/timing/total", total_time)

    def on_validation_epoch_start(self):
        """Clears CUDA cache before validation to reduce memory peak after training."""
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        total_samples = (
            self.val_classification_metrics.tp
            + self.val_classification_metrics.fp
            + self.val_classification_metrics.fn
            + self.val_classification_metrics.tn
        ).sum()

        if total_samples > 0:
            cls_metrics = self.val_classification_metrics.compute()
            cam_miou_metrics = self.val_cam_metrics.compute()
            miou_metrics = self.miou_metric.compute()

            self.val_classification_metrics.log_metrics(
                step=self.global_step, prefix="val/classification_", metrics=cls_metrics
            )
            self.val_cam_metrics.log_metrics(step=self.global_step, prefix="val/cam/")
            self.miou_metric.log_metrics(step=self.global_step, prefix="val/miou/")

            print(f"\n{'='*60}")
            print(f"EPOCH {self.current_epoch} - VALIDATION METRICS")
            print(f"{'='*60}")
            print(f"Classification: mAP={cls_metrics['mAP']:.4f} Accuracy={cls_metrics['macro']['accuracy']:.4f}")
            print(f"CAM mIoU: {cam_miou_metrics.get('mIoU', 0.0):.4f}")
            print(f"Seg mIoU: {miou_metrics.get('mIoU', 0.0):.4f}")
            print(f"{'='*60}\n")
        else:
            print(f"\nEPOCH {self.current_epoch} - No validation data processed\n")
            self.val_classification_metrics.log_metrics(step=self.global_step, prefix="val/classification_")

    def configure_optimizers(self):
        cfg = self.config
        if self.freeze_backbone:
            optimizer_groups = [
                {
                    "params": list(self.model1.classifier.parameters())
                    + list(self.model2.classifier.parameters())
                    + list(self.model1.decode_head.parameters())
                    + list(self.model2.decode_head.parameters()),
                    "lr": cfg.get("head_lr", 2e-5),
                    "weight_decay": cfg.get("head_weight_decay", 0.01),
                },
            ]
        else:
            optimizer_groups = [
                {
                    "params": list(self.model1.backbone.parameters()) + list(self.model2.backbone.parameters()),
                    "lr": cfg.get("backbone_lr", 1e-5),
                    "weight_decay": cfg.get("backbone_weight_decay", 0.01),
                },
                {
                    "params": list(self.model1.classifier.parameters())
                    + list(self.model2.classifier.parameters())
                    + list(self.model1.decode_head.parameters())
                    + list(self.model2.decode_head.parameters()),
                    "lr": cfg.get("head_lr", 2e-5),
                    "weight_decay": cfg.get("head_weight_decay", 0.01),
                },
            ]
        optimizer = AdamW(
            optimizer_groups,
            weight_decay=cfg.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("epochs", 50),
            eta_min=1e-7,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
