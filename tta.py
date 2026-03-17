"""
Test Time Augmentation para geração de pseudo-labels a partir de CAMs.
Uma única função unificada (param detach para compatibilidade com ambos os usos).
"""

import torch
import torch.nn.functional as F
from helpers.cam_processing import balance_class_presence


def generate_hard_pseudo_labels_tta(
    cams,
    y_cls,
    high_threshold,
    low_threshold,
    ignore_index,
    model,
    x,
    tta_weights=None,
    detach_cams=True,
):
    """
    Gera pseudo-labels com TTA (flip H/V, escalas 0.8/1.2, rotação 90/270).
    Combina CAMs com pesos e aplica balance_class_presence nas regiões de alta confiança.

    Args:
        cams: [B, C, H, W]
        y_cls: [B, num_classes] (com background no índice 0)
        high_threshold, low_threshold: limiares de confiança
        ignore_index: índice para pixels ignorados
        model: modelo para forward com TTA
        x: [B, C, H, W]
        tta_weights: lista de 7 pesos (default [0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
        detach_cams: se True, usa cams sem grad (treino); se False, mantém grad (val/debug)

    Returns:
        pseudo_labels: [B, H, W] long
    """
    if detach_cams:
        cams = cams.detach()
    B, C, H, W = cams.shape
    target_device = cams.device
    if tta_weights is None:
        tta_weights = [0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]
    all_cams = []
    try:
        cams_original = F.interpolate(cams, size=(H, W), mode='bilinear', align_corners=False)
        all_cams.append(cams_original)
        # Extrair só as CAMs e liberar o output completo imediatamente para evitar OOM
        with torch.amp.autocast('cuda', enabled=False):
            x_hflip = torch.flip(x, dims=[3])
            out_hflip = model(x_hflip, target_labels=y_cls, test=True)
            cams_hflip = torch.flip(out_hflip["cams"].detach(), dims=[3])
            del x_hflip, out_hflip
            torch.cuda.empty_cache()
            all_cams.append(F.interpolate(cams_hflip, size=(H, W), mode='bilinear', align_corners=False))
        with torch.amp.autocast('cuda', enabled=False):
            x_vflip = torch.flip(x, dims=[2])
            out_vflip = model(x_vflip, target_labels=y_cls, test=True)
            cams_vflip = torch.flip(out_vflip["cams"].detach(), dims=[2])
            del x_vflip, out_vflip
            torch.cuda.empty_cache()
            all_cams.append(F.interpolate(cams_vflip, size=(H, W), mode='bilinear', align_corners=False))
        with torch.amp.autocast('cuda', enabled=False):
            x_08 = F.interpolate(x, scale_factor=0.8, mode='bilinear', align_corners=False)
            out_08 = model(x_08, target_labels=y_cls, test=True)
            cams_08 = out_08["cams"].detach()
            del x_08, out_08
            torch.cuda.empty_cache()
            all_cams.append(F.interpolate(cams_08, size=(H, W), mode='bilinear', align_corners=False))
        with torch.amp.autocast('cuda', enabled=False):
            x_12 = F.interpolate(x, scale_factor=1.2, mode='bilinear', align_corners=False)
            out_12 = model(x_12, target_labels=y_cls, test=True)
            cams_12 = out_12["cams"].detach()
            del x_12, out_12
            torch.cuda.empty_cache()
            all_cams.append(F.interpolate(cams_12, size=(H, W), mode='bilinear', align_corners=False))
        with torch.amp.autocast('cuda', enabled=False):
            x_rot90 = torch.rot90(x, k=1, dims=[2, 3])
            out_rot90 = model(x_rot90, target_labels=y_cls, test=True)
            cams_rot90 = torch.rot90(out_rot90["cams"].detach(), k=-1, dims=[2, 3])
            del x_rot90, out_rot90
            torch.cuda.empty_cache()
            all_cams.append(F.interpolate(cams_rot90, size=(H, W), mode='bilinear', align_corners=False))
        with torch.amp.autocast('cuda', enabled=False):
            x_rot270 = torch.rot90(x, k=-1, dims=[2, 3])
            out_rot270 = model(x_rot270, target_labels=y_cls, test=True)
            cams_rot270 = torch.rot90(out_rot270["cams"].detach(), k=1, dims=[2, 3])
            del x_rot270, out_rot270
            torch.cuda.empty_cache()
            all_cams.append(F.interpolate(cams_rot270, size=(H, W), mode='bilinear', align_corners=False))
        for i, cam in enumerate(all_cams):
            if cam.shape != (B, C, H, W):
                all_cams[i] = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        weights = torch.tensor(tta_weights, device=target_device, dtype=torch.float32)
        weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        stacked = torch.stack(all_cams, dim=0).permute(1, 2, 3, 4, 0)
        mean_cams = (stacked * weights).sum(dim=4)
        mean_cams[:, 0] = 1 - torch.max(mean_cams[:, 1:], dim=1)[0]
        mean_cams = F.interpolate(mean_cams, size=(H, W), mode='bilinear', align_corners=False)
        del stacked, all_cams
        torch.cuda.empty_cache()
        target_mask = (y_cls > 0.1).float().unsqueeze(-1).unsqueeze(-1)
        mean_cams_masked = torch.where(
            target_mask > 0.5,
            mean_cams,
            torch.tensor(float('-inf'), device=mean_cams.device, dtype=mean_cams.dtype),
        )
        confidence, _ = torch.max(mean_cams_masked[:, 1:], dim=1)
        pseudo_labels = mean_cams_masked.argmax(dim=1)
        pseudo_labels[confidence < low_threshold] = 0
        pseudo_labels[(confidence >= low_threshold) & (confidence < high_threshold)] = ignore_index
        keep_mask = (confidence >= high_threshold).to(target_device)
        grouped_cams = []
        for cam_id in range(B):
            balanced_cams, _ = balance_class_presence(
                mean_cams[cam_id].cpu().detach(),
                y_cls[cam_id].cpu().detach(),
                min_fraction=0.03,
                factor=1.1,
                max_iter=10,
            )
            max_fg_balanced = torch.max(balanced_cams[1:], dim=0)[0]
            balanced_cams[0] = torch.clamp(1.0 - max_fg_balanced, min=0.0)
            y_cls_mask = y_cls[cam_id].cpu().detach().unsqueeze(1).unsqueeze(2)
            balanced_cams = torch.where(
                y_cls_mask > 0.5,
                balanced_cams,
                torch.tensor(float('-inf'), device=balanced_cams.device, dtype=balanced_cams.dtype),
            )
            grouped_cams.append(balanced_cams)
        grouped_cams = torch.stack(grouped_cams).to(target_device)
        pseudo_labels[keep_mask] = grouped_cams.argmax(dim=1)[keep_mask]
        return pseudo_labels
    finally:
        if 'all_cams' in locals():
            del all_cams
        if 'stacked' in locals():
            del stacked
        torch.cuda.empty_cache()
