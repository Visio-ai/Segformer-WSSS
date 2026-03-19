import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

def old_exclusivity_loss_vectorized(cams: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculates the exclusivity loss for the given CAMs and labels."""
    active_mask_bool = labels > 0.5
    active_mask = active_mask_bool.unsqueeze(-1).unsqueeze(-1)
    
    if not active_mask_bool.any():
        return torch.tensor(0.0, device=cams.device)
    
    cams_pos = F.relu(cams)
    cams_norm = cams_pos / (cams_pos.sum(dim=(2,3), keepdim=True) + 1e-8)
    cams_active = cams_norm * active_mask.float()
    
    batch, classes, H, W = cams.shape
    cams_flat = cams_active.flatten(2)
    similarity = torch.bmm(cams_flat, cams_flat.transpose(1,2))
    
    eye = torch.eye(classes, device=cams.device).unsqueeze(0)
    active_mask_2d = active_mask_bool.unsqueeze(2) & active_mask_bool.unsqueeze(1)
    pair_mask = (1 - eye) * active_mask_2d.float()
    
    penalty = torch.clamp(similarity, min=0)
    valid_pairs = pair_mask.sum()
    
    if valid_pairs > 0:
        loss = (penalty * pair_mask).sum() / valid_pairs
    else:
        loss = torch.tensor(0.0, device=cams.device)
    
    return loss



def loss_overlap(cams, labels=None):
    binary_maps = (cams > 0.5).float()

    if labels is not None:
        binary_maps = binary_maps * labels.unsqueeze(2).unsqueeze(2)

    overlap_map = torch.sum(binary_maps, dim=0)
    return torch.relu(overlap_map - 1).mean()


def loss_combined_exclusivity(cams, labels):
    return (
        loss_overlap(cams) + 
        old_exclusivity_loss_vectorized(cams[:,:,:,:], labels[:,:])
    )/ 2


def js_divergence(p, q, eps=1e-8):
    """Calcula a divergência de Jensen-Shannon entre duas distribuições."""
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    
    m = 0.5 * (p + q)
    
    kl_pm = F.kl_div(F.log_softmax(p, dim=-1), m, reduction='none').sum(dim=-1)
    kl_qm = F.kl_div(F.log_softmax(q, dim=-1), m, reduction='none').sum(dim=-1)
    
    return 0.5 * (kl_pm + kl_qm)


def improved_exclusivity_loss(cams, labels, temperature=1.0):
    """Versão melhorada com temperatura e normalização mais robusta"""
    active_mask = labels > 0.5
    if not active_mask.any():
        return torch.tensor(0.0, device=cams.device)
    
    # Aplicar softmax espacial para normalização mais suave
    cams_soft = F.softmax(cams / temperature, dim=1)
    cams_active = cams_soft * active_mask.unsqueeze(-1).unsqueeze(-1)
    
    # Calcular divergência de Jensen-Shannon entre pares
    batch, classes, H, W = cams.shape
    loss = 0.0
    count = 0
    
    for i in range(classes):
        for j in range(i+1, classes):
            if active_mask[:, i].any() and active_mask[:, j].any():
                cam_i = cams_active[:, i].flatten(1)
                cam_j = cams_active[:, j].flatten(1)
                
                # JS divergence é mais robusta que correlação
                js_div = js_divergence(cam_i, cam_j)
                loss += js_div.mean()
                count += 1
    
    return loss / max(count, 1)


def soft_overlap_loss(cams, labels, temperature=1.0):
    """Versão suave da perda de sobreposição"""
    active_mask = labels > 0.5
    if not active_mask.any():
        return torch.tensor(0.0, device=cams.device)
    
    # Softmax espacial em vez de threshold binário
    cams_soft = F.softmax(cams / temperature, dim=1)
    cams_active = cams_soft * active_mask.unsqueeze(-1).unsqueeze(-1)
    
    # Penalizar sobreposição suavemente
    overlap_map = torch.sum(cams_active, dim=1)
    # Penalizar quando a soma é > 1 (múltiplas classes ativas)
    penalty = F.relu(overlap_map - 1.0)
    
    return penalty.mean()


def optimized_exclusivity_loss(cams, labels, overlap_weight=0.7, similarity_weight=0.3, temperature=1.0):
    """
    Versão otimizada da perda de exclusividade com pesos adaptativos.
    
    Args:
        cams: Class Activation Maps [batch, classes, H, W]
        labels: Labels binários [batch, classes]
        overlap_weight: Peso para perda de sobreposição (default: 0.7)
        similarity_weight: Peso para perda de similaridade (default: 0.3)
        temperature: Temperatura para softmax (default: 1.0)
    
    Returns:
        Perda de exclusividade otimizada
    """
    overlap_loss = soft_overlap_loss(cams, labels, temperature)
    similarity_loss = improved_exclusivity_loss(cams, labels, temperature)
    
    return overlap_weight * overlap_loss + similarity_weight * similarity_loss


def multilabel_softmargin_loss_with_focus(pred, target, focus_weights=None, reduction='mean'):
    """
    MultiLabelSoftMarginLoss com suporte a pesos de foco.
    
    Args:
        pred: logits de predição [batch_size, num_classes]
        target: targets binários [batch_size, num_classes] 
        focus_weights: pesos de foco por exemplo [batch_size] (default: None)
        reduction: tipo de redução ('mean', 'sum', 'none')
    
    Returns:
        MultiLabelSoftMarginLoss calculado com pesos de foco
    """
    # Calcula MultiLabelSoftMarginLoss por exemplo
    loss_per_sample = F.multilabel_soft_margin_loss(pred, target, reduction='none')
    
    # Aplica pesos de foco se fornecidos
    if focus_weights is not None:
        loss_per_sample = loss_per_sample * focus_weights
    
    if reduction == 'mean':
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:
        return loss_per_sample


def discrepancy_loss(f1, f2, eps=1e-8):
    """
    Calcula a perda de discrepância entre dois feature maps.
    
    Args:
        f1, f2: feature maps [B, D, H, W]
        eps: epsilon para estabilidade numérica
    
    Returns:
        Perda de discrepância baseada em similaridade de cosseno
    """
    f1_flat = f1.view(f1.shape[0], -1)
    f2_flat = f2.view(f2.shape[0], -1)

    cos_simi = nn.CosineSimilarity(dim=-1, eps=1e-6)
    sim_loss_1 = 1 + cos_simi(f1_flat.detach(), f2_flat).mean()
    sim_loss_2 = 1 + cos_simi(f2_flat.detach(), f1_flat).mean()

    return sim_loss_1 + sim_loss_2


def compute_equivariance_loss(model, x, y_cls, original_cams):
    """
    Calcula a perda de equivariância aplicando transformações de data augmentation.
    
    Args:
        model: modelo para inferência
        x: tensor de entrada [B, C, H, W]
        y_cls: labels de classificação [B, num_classes]
        original_cams: CAMs originais [B, num_classes, H, W]
    
    Returns:
        Perda de equivariância combinando MSE dos CAMs e perda de classificação
    """
    # Aplica transformações de cor em uma cópia para não modificar o tensor original
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    x_aug = color_jitter(x)

    # Escolhe e aplica transformações geométricas
    flip_x = torch.rand(1).item() < 0.5
    flip_y = torch.rand(1).item() < 0.5
    if flip_x:
        x_aug = x_aug.flip(dims=[3])
    if flip_y:
        x_aug = x_aug.flip(dims=[2])

    # Redimensiona a imagem com um fator aleatório
    batch_size_factor = np.random.uniform(1.0, 1.2)
    new_size = int(x_aug.shape[2] * batch_size_factor)
    x_aug = F.interpolate(x_aug, size=(new_size, new_size), mode='bilinear', align_corners=False)

    # Faz a inferência no input aumentado
    outputs = model(x_aug, target_labels=y_cls, test=True)
    aug_cams = outputs["cams"]
    cls_loss = multilabel_softmargin_loss_with_focus(
        outputs["classification"][:,:], y_cls[:,:]
    )
    # Libera a memória dos tensores que não serão mais usados
    del outputs, x_aug
    torch.cuda.empty_cache()

    # Aplica a transformação geométrica inversa nos CAMs resultantes
    if flip_y:
        aug_cams = aug_cams.flip(dims=[2])
    if flip_x:
        aug_cams = aug_cams.flip(dims=[3])

    aug_cams = F.interpolate(aug_cams, size=original_cams.shape[2:], mode='bilinear', align_corners=False)
    
    loss = (F.mse_loss(original_cams, aug_cams.detach()) + cls_loss ) /2
    
    del aug_cams
    torch.cuda.empty_cache()
    
    return loss


def compute_equivariance_loss_segmentation(model, x, y_cls, original_segmentation):
    """
    Calcula a perda de equivariância para segmentação aplicando transformações de data augmentation.
    
    Args:
        model: modelo para inferência
        x: tensor de entrada [B, C, H, W]
        y_cls: labels de classificação [B, num_classes]
        original_segmentation: segmentação original [B, num_classes, H, W]
    
    Returns:
        Perda de equivariância combinando MSE da segmentação e perda de classificação
    """
    # Aplica transformações de cor em uma cópia para não modificar o tensor original
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    x_aug = color_jitter(x)

    # Escolhe e aplica transformações geométricas
    flip_x = torch.rand(1).item() < 0.5
    flip_y = torch.rand(1).item() < 0.5
    if flip_x:
        x_aug = x_aug.flip(dims=[3])
    if flip_y:
        x_aug = x_aug.flip(dims=[2])

    # Redimensiona a imagem com um fator aleatório
    batch_size_factor = np.random.uniform(1.0, 1.2)
    new_size = int(x_aug.shape[2] * batch_size_factor)
    x_aug = F.interpolate(x_aug, size=(new_size, new_size), mode='bilinear', align_corners=False)

    # Faz a inferência no input aumentado
    outputs = model(x_aug, target_labels=[], test=True)
    aug_segmentation = outputs["segmentation"]
    cls_loss = multilabel_softmargin_loss_with_focus(
        outputs["classification"][:,:], y_cls[:,:]
    )
    # Libera a memória dos tensores que não serão mais usados
    del outputs, x_aug
    torch.cuda.empty_cache()

    # Aplica a transformação geométrica inversa na segmentação resultante
    if flip_y:
        aug_segmentation = aug_segmentation.flip(dims=[2])
    if flip_x:
        aug_segmentation = aug_segmentation.flip(dims=[3])

    aug_segmentation = F.interpolate(aug_segmentation, size=original_segmentation.shape[2:], mode='bilinear', align_corners=False)
    
    # Aplica sigmoid na segmentação aumentada para ter valores entre 0 e 1
    aug_segmentation = torch.sigmoid(aug_segmentation)
    
    loss = (F.mse_loss(original_segmentation, aug_segmentation.detach()) + cls_loss) / 2
    
    del aug_segmentation
    torch.cuda.empty_cache()
    
    return loss
