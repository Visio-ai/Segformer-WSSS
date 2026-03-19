"""
Augmentations for WSSS: CutMix and MixUp with spatial masks.
Labels are [B, num_classes] with background at index 0.
"""

import numpy as np
import torch


def cutmix_with_masks(x1, y1, x2, y2, alpha=1.0):
    """
    CutMix between two batches; returns spatial mask and class lists per image.
    y1, y2: [B, num_classes] with background at index 0.
    """
    B, C, H, W = x1.shape
    device = x1.device
    lam = np.random.beta(alpha, alpha)
    lam = max(0.0, min(1.0, lam))
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mask_spatial = torch.ones(B, H, W, device=device)
    mask_spatial[:, bby1:bby2, bbx1:bbx2] = 0.0
    x_mixed = x1.clone()
    x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
    y_mixed = torch.logical_or(y1 > 0.5, y2 > 0.5).float()
    y1_classes = []
    y2_classes = []
    for b in range(B):
        y1_fg = y1[b, 1:]
        y1_classes.append(torch.where(y1_fg > 0.1)[0].cpu().tolist())
        y2_fg = y2[b, 1:]
        y2_classes.append(torch.where(y2_fg > 0.1)[0].cpu().tolist())
    return x_mixed, y_mixed, mask_spatial, y1_classes, y2_classes


def mixup_with_masks(x1, y1, x2, y2, alpha=0.2):
    """
    MixUp between two batches; returns mixing lambda and class lists per image.
    y1, y2: [B, num_classes] with background at index 0.
    """
    B = x1.shape[0]
    device = x1.device
    lam = np.random.beta(alpha, alpha, size=B) if alpha > 0 else np.ones(B)
    lam = np.clip(lam, 0.0, 1.0)
    lam_tensor = torch.from_numpy(lam).float().to(device).view(B, 1, 1, 1)
    x_mixed = lam_tensor * x1 + (1.0 - lam_tensor) * x2
    lam_batch = torch.from_numpy(lam).float().to(device).view(B, 1)
    y_mixed = (lam_batch * y1 + (1.0 - lam_batch) * y2 > 0.5).float()
    y1_classes = []
    y2_classes = []
    for b in range(B):
        y1_classes.append(torch.where(y1[b, 1:] > 0.1)[0].cpu().tolist())
        y2_classes.append(torch.where(y2[b, 1:] > 0.1)[0].cpu().tolist())
    lambda_mix = torch.from_numpy(lam).float().to(device)
    return x_mixed, y_mixed, lambda_mix, y1_classes, y2_classes
