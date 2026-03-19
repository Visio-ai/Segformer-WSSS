import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def apply_ignore_index_to_cams(cams, masks, ignore_index=-1):
    """
    Aplica ignore index nas CAMs baseado nas máscaras de background artificial
    
    Args:
        cams: torch.Tensor (B, C, H, W) - CAMs do modelo
        masks: torch.Tensor (B, H, W) - Máscaras indicando regiões válidas (1) vs background artificial (0)
        ignore_index: int - Valor para ignore index
    
    Returns:
        cams_with_ignore: torch.Tensor (B, C, H, W) - CAMs com ignore index aplicado
    """
    # Expandir máscara para todas as classes
    if masks.dim() == 3:  # (B, H, W)
        masks = masks.unsqueeze(1)  # (B, 1, H, W)
    
    # Expandir para todas as classes se necessário
    if masks.shape[1] == 1 and cams.shape[1] > 1:
        masks = masks.expand(-1, cams.shape[1], -1, -1)  # (B, C, H, W)
    
    # Converter máscaras para boolean (True = válido, False = ignore)
    valid_regions = masks > 0.5
    
    # Aplicar ignore index onde a máscara é 0 (background artificial)
    cams_with_ignore = cams.clone()
    cams_with_ignore[~valid_regions] = ignore_index
    
    return cams_with_ignore


def apply_ignore_index_to_segmentation(seg_masks, masks, ignore_index=-1):
    """
    Aplica ignore index nas máscaras de segmentação baseado nas máscaras de background artificial
    
    Args:
        seg_masks: torch.Tensor (B, H, W) - Máscaras de segmentação
        masks: torch.Tensor (B, H, W) - Máscaras indicando regiões válidas (1) vs background artificial (0)
        ignore_index: int - Valor para ignore index
    
    Returns:
        seg_masks_with_ignore: torch.Tensor (B, H, W) - Máscaras de segmentação com ignore index aplicado
    """
    # Converter máscaras para boolean (True = válido, False = ignore)
    valid_regions = masks > 0.5
    
    # Aplicar ignore index onde a máscara é 0 (background artificial)
    seg_masks_with_ignore = seg_masks.clone()
    seg_masks_with_ignore[~valid_regions] = ignore_index
    
    return seg_masks_with_ignore


def masked_bce_loss_with_ignore_index(predictions, targets, masks, ignore_index=-1):
    """
    Calcula BCE loss aplicando ignore index nas áreas de background artificial
    
    Args:
        predictions: torch.Tensor (B, C, H, W) - Predições do modelo
        targets: torch.Tensor (B, C, H, W) - Targets
        masks: torch.Tensor (B, H, W) - Máscaras indicando regiões válidas (1) vs background artificial (0)
        ignore_index: int - Valor para ignore index
    
    Returns:
        loss: torch.Tensor - Loss calculada apenas nas regiões válidas
    """
    # Expandir máscara para todas as classes
    if masks.dim() == 3:  # (B, H, W)
        masks = masks.unsqueeze(1)  # (B, 1, H, W)
    
    # Aplicar máscara nas predições e targets
    masked_predictions = predictions * masks
    masked_targets = targets * masks
    
    # Calcular BCE loss apenas nas regiões válidas
    bce_loss = F.binary_cross_entropy_with_logits(
        masked_predictions, 
        masked_targets, 
        reduction='none'
    )
    
    # Aplicar máscara na loss (ignorar regiões de background artificial)
    masked_loss = bce_loss * masks
    
    # Calcular loss média apenas nas regiões válidas
    valid_pixels = masks.sum()
    if valid_pixels > 0:
        loss = masked_loss.sum() / valid_pixels
    else:
        loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    return loss


def masked_dice_loss_with_ignore_index(predictions, targets, masks, ignore_index=-1, smooth=1e-6):
    """
    Calcula Dice loss aplicando ignore index nas áreas de background artificial
    
    Args:
        predictions: torch.Tensor (B, C, H, W) - Predições do modelo
        targets: torch.Tensor (B, C, H, W) - Targets
        masks: torch.Tensor (B, H, W) - Máscaras indicando regiões válidas (1) vs background artificial (0)
        ignore_index: int - Valor para ignore index
        smooth: float - Smoothing factor
    
    Returns:
        loss: torch.Tensor - Dice loss calculada apenas nas regiões válidas
    """
    # Expandir máscara para todas as classes
    if masks.dim() == 3:  # (B, H, W)
        masks = masks.unsqueeze(1)  # (B, 1, H, W)
    
    # Aplicar sigmoid nas predições
    predictions = torch.sigmoid(predictions)
    
    # Aplicar máscara nas predições e targets
    masked_predictions = predictions * masks
    masked_targets = targets * masks
    
    # Calcular interseção e união apenas nas regiões válidas
    intersection = (masked_predictions * masked_targets).sum(dim=(2, 3))
    union = masked_predictions.sum(dim=(2, 3)) + masked_targets.sum(dim=(2, 3))
    
    # Calcular Dice score
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Calcular loss (1 - dice)
    dice_loss = 1.0 - dice
    
    # Calcular loss média apenas nas regiões válidas
    valid_pixels = masks.sum(dim=(2, 3))
    valid_classes = (valid_pixels > 0).float()
    
    if valid_classes.sum() > 0:
        loss = (dice_loss * valid_classes).sum() / valid_classes.sum()
    else:
        loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    return loss


def create_ignore_mask_from_cams(cams, threshold=0.1):
    """
    Cria máscara de ignore baseada nas CAMs (áreas com baixa ativação)
    
    Args:
        cams: torch.Tensor (B, H, W) - CAMs do modelo
        threshold: float - Threshold para considerar região como background
    
    Returns:
        ignore_mask: torch.Tensor (B, H, W) - Máscara de ignore (1 = válido, 0 = ignore)
    """
    # Normalizar CAMs
    cams_normalized = F.softmax(cams, dim=-1)
    
    # Criar máscara de ignore baseada no threshold
    ignore_mask = (cams_normalized > threshold).float()
    
    return ignore_mask


def combined_mask_loss(predictions, targets, cams, masks, 
                      bce_weight=1.0, dice_weight=1.0, cam_weight=0.5,
                      ignore_index=-1):
    """
    Combina BCE loss, Dice loss e CAM loss com ignore index
    
    Args:
        predictions: torch.Tensor (B, C, H, W) - Predições do modelo
        targets: torch.Tensor (B, C, H, W) - Targets
        cams: torch.Tensor (B, H, W) - CAMs do modelo
        masks: torch.Tensor (B, H, W) - Máscaras indicando regiões válidas
        bce_weight: float - Peso para BCE loss
        dice_weight: float - Peso para Dice loss
        cam_weight: float - Peso para CAM loss
        ignore_index: int - Valor para ignore index
    
    Returns:
        total_loss: torch.Tensor - Loss combinada
        loss_dict: dict - Dicionário com losses individuais
    """
    # BCE loss com ignore index
    bce_loss = masked_bce_loss_with_ignore_index(predictions, targets, masks, ignore_index)
    
    # Dice loss com ignore index
    dice_loss = masked_dice_loss_with_ignore_index(predictions, targets, masks, ignore_index)
    
    # CAM loss (se cams for fornecido)
    cam_loss = torch.tensor(0.0, device=predictions.device)
    if cams is not None:
        # Aplicar ignore index nas CAMs
        cams_with_ignore = apply_ignore_index_to_cams(cams, masks, ignore_index)
        
        # Calcular loss das CAMs (exemplo: MSE com targets de segmentação)
        if targets.dim() == 4:  # (B, C, H, W)
            # Reduzir targets para (B, H, W) se necessário
            targets_2d = targets.sum(dim=1)  # Soma das classes
        else:
            targets_2d = targets
            
        cam_loss = F.mse_loss(cams_with_ignore, targets_2d)
    
    # Loss total
    total_loss = bce_weight * bce_loss + dice_weight * dice_loss + cam_weight * cam_loss
    
    loss_dict = {
        'total_loss': total_loss,
        'bce_loss': bce_loss,
        'dice_loss': dice_loss,
        'cam_loss': cam_loss
    }
    
    return total_loss, loss_dict
