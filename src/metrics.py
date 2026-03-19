"""
Module for computing WSSS semantic segmentation metrics.
"""

import torch
import numpy as np


class MiouMetric:
    """
    Manually computes mIoU pixel by pixel.

    This class computes per-class IoU and mean IoU (mIoU) manually,
    comparing predictions and ground truth pixel by pixel.

    Args:
        num_classes (int): Total number of classes.
        ignore_index (int): Index to be ignored in predictions/ground truth.
        logger (Optional[Logger]): A PyTorch Lightning logger for automatic logging.
    """
    
    def __init__(self, num_classes, ignore_index=255, logger=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.logger = logger
        self.reset()
    
    def reset(self):
        """Resets the counters."""
        # Per-class counters: TP, FP, FN
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
    
    def update(self, pred_indices, target_indices):
        """
        Updates the counters with a new batch of predictions and targets.

        Args:
            pred_indices (torch.Tensor): Predictions as class indices [B, H, W]
            target_indices (torch.Tensor): Ground truth as class indices [B, H, W]
        """
        # Move to CPU if needed
        pred_indices = pred_indices.cpu()
        target_indices = target_indices.cpu()
        
        # Flatten for pixel-wise processing
        pred_flat = pred_indices.flatten()  # [B*H*W]
        target_flat = target_indices.flatten()  # [B*H*W]
        
        # Create mask for valid pixels (not ignored)
        valid_mask = (target_flat != self.ignore_index) & (pred_flat != self.ignore_index)
        
        # Filter only valid pixels
        valid_preds = pred_flat[valid_mask]
        valid_targets = target_flat[valid_mask]
        
        # Ensure indices are within valid range
        valid_preds = torch.clamp(valid_preds, 0, self.num_classes - 1)
        valid_targets = torch.clamp(valid_targets, 0, self.num_classes - 1)
        
        # Compute TP, FP, FN per class
        for c in range(self.num_classes):
            # True Positives: pixels where pred == target == c
            tp = ((valid_preds == c) & (valid_targets == c)).sum().item()
            
            # False Positives: pixels where pred == c but target != c
            fp = ((valid_preds == c) & (valid_targets != c)).sum().item()
            
            # False Negatives: pixels where target == c but pred != c
            fn = ((valid_targets == c) & (valid_preds != c)).sum().item()
            
            self.tp[c] += tp
            self.fp[c] += fp
            self.fn[c] += fn
    
    def compute(self):
        """
        Computes IoU per class and mIoU.

        Returns:
            dict: Dictionary with IoU per class and mIoU.
        """
        metrics = {}
        
        # Compute IoU per class
        iou_per_class = []
        for c in range(self.num_classes):
            # IoU = TP / (TP + FP + FN)
            iou = self.tp[c] / (self.tp[c] + self.fp[c] + self.fn[c] + 1e-8)
            iou_per_class.append(iou.item())
            metrics[f'IoU/class_{c}'] = iou.item()
        
        # Compute mIoU (mean of IoUs)
        miou = np.mean(iou_per_class)
        metrics['mIoU'] = miou
        
        return metrics
    
    def log_metrics(self, prefix="", step=None):
        """Logs the computed metrics using the provided logger."""
        metrics = self.compute()
        if self.logger:
            for name, value in metrics.items():
                # LightningModule.log() does not accept step as a named argument
                self.logger.log(f"{prefix}{name}", value)

