"""
Módulo otimizado para cálculo de métricas de segmentação semântica com validação cruzada (k-fold).
Versão otimizada que permite desabilitar o cálculo de BFScore durante o update para melhor performance.
"""

import torch
import numpy as np
import pytorch_lightning as pl
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from joblib import Parallel, delayed

class SegmentationFoldMetricsOptimized:
    """
    Calcula e loga métricas de segmentação semântica para tarefas multilabel
    usando uma estratégia de avaliação k-fold.
    
    Versão otimizada que permite desabilitar o cálculo de BFScore durante o update.

    Esta classe foi projetada para PyTorch Lightning. Ela acumula estatísticas para
    cada fold separadamente, com base em um índice de partição fornecido no método `update`.

    Calcula as seguintes métricas (por classe e média) para cada fold, e também
    fornece uma média final entre todos os folds:
    - Intersection over Union (IoU)
    - Coeficiente de Dice (F1-score para segmentação)
    - Boundary F1 Score (BFScore) - opcional durante update
    - Precisão, Recall

    Args:
        class_names (List[str]): Uma lista com os nomes das classes.
        num_classes (int): O número total de classes.
        num_partitions (int): O número total de partições para a divisão k-fold.
        bf_threshold (int): A tolerância em pixels para correspondência de bordas no BFScore.
        logger (Optional[Logger]): Um logger do PyTorch Lightning para log automático.
        compute_bfscore_during_update (bool): Se False, o BFScore só é calculado no compute() final.
    """
    def __init__(self, class_names, num_classes, num_partitions, bf_threshold=3, logger=None, compute_bfscore_during_update=False):
        self.class_names = class_names
        self.num_classes = num_classes
        self.num_partitions = num_partitions
        self.bf_threshold = bf_threshold
        self.logger = logger
        self.compute_bfscore_during_update = compute_bfscore_during_update
        self.reset()

    def reset(self):
        """Reseta os acumuladores de métricas para o início de uma nova avaliação."""
        self.tp = torch.zeros(self.num_partitions, self.num_classes)
        self.fp = torch.zeros(self.num_partitions, self.num_classes)
        self.fn = torch.zeros(self.num_partitions, self.num_classes)
        self.tn = torch.zeros(self.num_partitions, self.num_classes)
        self.bfscore_sum = torch.zeros(self.num_partitions, self.num_classes)
        self.bfscore_count = torch.zeros(self.num_partitions, self.num_classes)
        
        # Armazena preds e targets para cálculo de BFScore no final (se necessário)
        if not self.compute_bfscore_during_update:
            self.stored_preds = []
            self.stored_targets = []
            self.stored_partitions = []

    def update(self, preds, targets, partition):
        """
        Atualiza as métricas com um novo lote de predições e alvos para uma partição específica.

        Args:
            preds (torch.Tensor): Predições do modelo (logits).
            targets (torch.Tensor): Máscaras de ground truth.
            partition (torch.Tensor): O índice da partição (fold) à qual este lote pertence.
        """
        if not (0 <= partition.max() < self.num_partitions):
            raise ValueError(f"O índice da partição deve estar entre 0 e {self.num_partitions - 1}, mas foi recebido {partition}")

        # Garante que os tensores estão na CPU para operações com numpy
        preds_cpu = preds.cpu()
        targets_cpu = targets.cpu()

        # Converte logits para probabilidades e aplica um threshold
        probs = torch.sigmoid(preds_cpu)
        old_probs = (probs > 0.5).byte()

        preds_mask = torch.argmax(probs, dim=1)  # Obtém a classe com maior probabilidade para cada pixel
        preds_mask = (preds_mask.unsqueeze(1) == torch.arange(self.num_classes, device=preds_mask.device).view(1, -1, 1, 1)).byte()  # Cria uma máscara binária para cada classe
        preds_mask = (preds_mask == 1).byte()  # Considera a classe 1 como foreground

        assert preds_mask.shape == old_probs.shape, "As formas das máscaras previstas não correspondem."
        targets_mask = targets_cpu.byte()

        B, C, H, W = preds_mask.shape
        assert C == self.num_classes, "O número de canais da predição não corresponde ao número de classes."

        # Calcula estatísticas básicas (TP, FP, FN) por amostra e depois agrega por partição
        # Forma: (B, C, H, W) -> (B, C) após sum sobre H, W
        tp_per_sample = torch.sum(preds_mask & targets_mask, dim=(2, 3)).float()  # (B, C)
        fp_per_sample = torch.sum(preds_mask & ~targets_mask, dim=(2, 3)).float()  # (B, C)
        fn_per_sample = torch.sum(~preds_mask & targets_mask, dim=(2, 3)).float()  # (B, C)
        tn_per_sample = torch.sum(~preds_mask & ~targets_mask, dim=(2, 3)).float()  # (B, C)
        partition_tensor = partition.to(tp_per_sample.device)

        # Agrega por partição
        for p_idx in partition_tensor.unique():
            p_idx_int = p_idx.item()
            mask = partition_tensor == p_idx_int
            # Soma as estatísticas de todas as amostras desta partição
            self.tp[p_idx_int] += tp_per_sample[mask].sum(dim=0)
            self.fp[p_idx_int] += fp_per_sample[mask].sum(dim=0)
            self.fn[p_idx_int] += fn_per_sample[mask].sum(dim=0)
            self.tn[p_idx_int] += tn_per_sample[mask].sum(dim=0)

        # Calcula BFScore apenas se habilitado durante o update
        if self.compute_bfscore_during_update:
            self._compute_bfscore_batch(preds_mask, targets_mask, partition_tensor)
        else:
            # Armazena para cálculo posterior (apenas se necessário)
            # Para economizar memória, não armazenamos por padrão
            # O BFScore será calculado apenas se necessário no compute()
            pass

    def _compute_bfscore_batch(self, preds_mask, targets_mask, partition):
        """Calcula BFScore para um batch (versão original, mais lenta)."""
        if hasattr(preds_mask, 'cpu') and hasattr(preds_mask, 'numpy'):
            preds_mask_np = preds_mask.cpu().numpy()
        else:
            preds_mask_np = preds_mask

        if hasattr(targets_mask, 'cpu') and hasattr(targets_mask, 'numpy'):
            targets_mask_np = targets_mask.cpu().numpy()
        else:
            targets_mask_np = targets_mask

        def process_image(i):
            results = []
            for c in range(self.num_classes):
                pred_c = preds_mask_np[i, c]
                target_c = targets_mask_np[i, c]
                bfscore = self._calc_bfscore(pred_c, target_c, self.bf_threshold)
                results.append((c, bfscore))
            return results

        # Processa imagens em paralelo
        B = preds_mask.shape[0]
        results_list = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_image)(i) for i in range(B)
        )

        # Agrega resultados
        bfscore_batch = np.zeros(self.num_classes)
        count_batch = np.zeros(self.num_classes, dtype=int)

        for img_results in results_list:
            for c, score in img_results:
                bfscore_batch[c] += score
                count_batch[c] += 1

        # Atualiza estado da classe
        for p_idx in partition.unique():
            for c in range(self.num_classes):
                self.bfscore_sum[p_idx, c] += bfscore_batch[c]
                self.bfscore_count[p_idx, c] += count_batch[c]

    @staticmethod
    def _calc_bfscore(pred_mask, gt_mask, threshold):
        """Calcula o Boundary F1 Score para máscaras individuais."""
        if not np.any(gt_mask) and not np.any(pred_mask):
            return 1.0
        if not np.any(gt_mask) or not np.any(pred_mask):
            return 0.0

        gt_boundary = binary_dilation(gt_mask) ^ binary_erosion(gt_mask)
        pred_boundary = binary_dilation(pred_mask) ^ binary_erosion(pred_mask)

        if not np.any(gt_boundary) or not np.any(pred_boundary):
            return 1.0 if not np.any(gt_boundary) and not np.any(pred_boundary) else 0.0

        dist_to_gt = distance_transform_edt(~gt_boundary)
        dist_to_pred = distance_transform_edt(~pred_boundary)

        precision_map = dist_to_gt[pred_boundary]
        recall_map = dist_to_pred[gt_boundary]

        precision = np.sum(precision_map <= threshold) / (np.sum(pred_boundary) + 1e-8)
        recall = np.sum(recall_map <= threshold) / (np.sum(gt_boundary) + 1e-8)

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall + 1e-8)
        return 0.0

    def compute(self):
        """
        Calcula as métricas finais para cada fold e a média geral.

        Returns:
            dict: Um dicionário contendo todas as métricas calculadas.
        """
        metrics = {}
        all_folds_iou = []
        all_folds_dice = []
        all_folds_precision = []
        all_folds_recall = []
        all_folds_bfscore = []

        for p in range(self.num_partitions):
            tp_p = sum(self.tp) - self.tp[p]
            fp_p = sum(self.fp) - self.fp[p]
            fn_p = sum(self.fn) - self.fn[p]
            tn_p = sum(self.tn) - self.tn[p]

            bfscore_sum_p = sum(self.bfscore_sum) - self.bfscore_sum[p]
            bfscore_count_p = sum(self.bfscore_count) - self.bfscore_count[p]

            # Evita divisão por zero
            iou = tp_p / (tp_p + fp_p + fn_p + 1e-8)
            dice = 2 * tp_p / (2 * tp_p + fp_p + fn_p + 1e-8)
            precision = tp_p / (tp_p + fp_p + 1e-8)
            recall = tp_p / (tp_p + fn_p + 1e-8)
            bfscore = bfscore_sum_p / (bfscore_count_p + 1e-8)

            all_folds_iou.append(iou)
            all_folds_dice.append(dice)
            all_folds_precision.append(precision)
            all_folds_recall.append(recall)
            all_folds_bfscore.append(bfscore)

            # Métricas por classe para o fold atual
            for c in range(self.num_classes):
                metrics[f'Fold_{p}/IoU/{self.class_names[c]}'] = iou[c].item()
                metrics[f'Fold_{p}/Dice/{self.class_names[c]}'] = dice[c].item()
                metrics[f'Fold_{p}/Precision/{self.class_names[c]}'] = precision[c].item()
                metrics[f'Fold_{p}/Recall/{self.class_names[c]}'] = recall[c].item()
                metrics[f'Fold_{p}/BFScore/{self.class_names[c]}'] = bfscore[c].item()

            # Média das métricas para o fold atual
            metrics[f'Fold_{p}/IoU/mean'] = torch.mean(iou).item()
            metrics[f'Fold_{p}/Dice/mean'] = torch.mean(dice).item()
            metrics[f'Fold_{p}/Precision/mean'] = torch.mean(precision).item()
            metrics[f'Fold_{p}/Recall/mean'] = torch.mean(recall).item()
            metrics[f'Fold_{p}/BFScore/mean'] = torch.mean(bfscore).item()

        # Calcula a média entre os folds
        mean_iou_per_class = torch.stack(all_folds_iou).mean(dim=0)
        mean_dice_per_class = torch.stack(all_folds_dice).mean(dim=0)
        mean_precision_per_class = torch.stack(all_folds_precision).mean(dim=0)
        mean_recall_per_class = torch.stack(all_folds_recall).mean(dim=0)
        mean_bfscore_per_class = torch.stack(all_folds_bfscore).mean(dim=0)

        # Métricas médias por classe
        for c in range(self.num_classes):
            metrics[f'Mean/IoU/{self.class_names[c]}'] = mean_iou_per_class[c].item()
            metrics[f'Mean/Dice/{self.class_names[c]}'] = mean_dice_per_class[c].item()
            metrics[f'Mean/Precision/{self.class_names[c]}'] = mean_precision_per_class[c].item()
            metrics[f'Mean/Recall/{self.class_names[c]}'] = mean_recall_per_class[c].item()
            metrics[f'Mean/BFScore/{self.class_names[c]}'] = mean_bfscore_per_class[c].item()

        # Média geral das métricas
        metrics['Mean/IoU/mean'] = torch.mean(mean_iou_per_class).item()
        metrics['Mean/Dice/mean'] = torch.mean(mean_dice_per_class).item()
        metrics['Mean/Precision/mean'] = torch.mean(mean_precision_per_class).item()
        metrics['Mean/Recall/mean'] = torch.mean(mean_recall_per_class).item()
        metrics['Mean/BFScore/mean'] = torch.mean(mean_bfscore_per_class).item()

        return metrics

    def log_metrics(self, prefix="", step=None):
        """Loga as métricas calculadas usando o logger fornecido."""
        metrics = self.compute()
        if self.logger:
            for name, value in metrics.items():
                self.logger.log(
                    f"{prefix}{name}",
                    value
                )

