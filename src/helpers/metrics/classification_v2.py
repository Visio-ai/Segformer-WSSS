import torch
import numpy as np
from torchmetrics import AveragePrecision
from typing import Optional, List, Dict
from lightning.pytorch.loggers import Logger

class MultilabelMetrics:
    """
    Calculates and logs multilabel classification metrics.

    This class is designed to be used within a PyTorch Lightning training loop.
    It accumulates statistics over batches and calculates key metrics such as
    mean Average Precision (mAP), macro-averaged F1-score, precision, and recall,
    as well as their per-class equivalents.

    The class does not store predictions and targets in memory to avoid high
    memory consumption, instead accumulating confusion matrix components (TP, FP, FN, TN)
    and using `torchmetrics` for mAP calculation.

    Args:
        class_names (List[str]): A list of strings representing the names of the classes.
            The order must correspond to the output channels of the model.
        logger (Optional[Logger]): A PyTorch Lightning logger instance (e.g., TensorBoardLogger)
            to automatically log computed metrics. If None, logging is disabled.
        threshold (float): The confidence threshold (0.0 to 1.0) to binarize
            predictions for calculating F1, precision, and recall. Default is 0.5.

    Example:
        >>> class_names = ['cat', 'dog', 'bird']
        >>> metrics = MultilabelMetrics(class_names)
        >>> for batch in dataloader:
        ...     preds, targets = model(batch)
        ...     metrics.update(preds, targets)
        >>>
        >>> epoch_metrics = metrics.compute()
        >>> print(epoch_metrics['macro']['f1'])
        >>> metrics.reset() # Reset for the next epoch
    """
    def __init__(self, class_names: List[str], logger: Optional[Logger] = None, threshold: float = 0.5):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.threshold = threshold
        self.logger = logger
        
        # Use torchmetrics for mAP calculation
        if self.num_classes == 1:
            self.ap_metric = AveragePrecision(task='binary')
        else:
            self.ap_metric = AveragePrecision(task='multilabel', num_labels=self.num_classes)
            
        self.reset()
    
    def reset(self):
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.tn = torch.zeros(self.num_classes)
        self.ap_metric.reset()
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.cpu().float()
        targets = targets.cpu().int()
        
        # Update mAP metric
        self.ap_metric.update(preds, targets)
        
        # Update confusion matrix components
        binary_preds = (preds >= self.threshold).int()
        
        self.tp += torch.sum((binary_preds == 1) & (targets == 1), dim=0)
        self.fp += torch.sum((binary_preds == 1) & (targets == 0), dim=0)
        self.fn += torch.sum((binary_preds == 0) & (targets == 1), dim=0)
        self.tn += torch.sum((binary_preds == 0) & (targets == 0), dim=0)

    def compute(self) -> Dict:
        try:
            mAP = self.ap_metric.compute()
        except:
            mAP = torch.tensor(0.0)
            print("Error computing mAP. Ensure predictions and targets are correctly formatted.")
        
        # Calculate accuracy
        total_samples = self.tp + self.fp + self.fn + self.tn
        accuracy_per_class = (self.tp + self.tn) / (total_samples + 1e-12)
        macro_accuracy = torch.mean(accuracy_per_class).item()
        
        precision_per_class = self.tp / (self.tp + self.fp + 1e-12)
        recall_per_class = self.tp / (self.tp + self.fn + 1e-12)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-12)
        
        macro_precision = torch.mean(precision_per_class).item()
        macro_recall = torch.mean(recall_per_class).item()
        macro_f1 = torch.mean(f1_per_class).item()
        
        return {
            'mAP': mAP.item(),
            'macro': {
                'accuracy': macro_accuracy,
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'per_class': {
                'accuracy': accuracy_per_class.tolist(),
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist(),
                'tp': self.tp.tolist(),
                'fp': self.fp.tolist(),
                'fn': self.fn.tolist(),
                'tn': self.tn.tolist()
            }
        }
    
    def log_metrics(self, step: int, prefix: str = "val/", metrics: Optional[Dict] = None):
        """
        Log métricas com timestamp e contexto estruturado.
        
        Args:
            step: Step number for logging
            prefix: Prefix for metric keys
            metrics: Optional pre-computed metrics dict. If None, will call compute().
        """
        import time
        timestamp = time.time()  # Usar timestamp numérico
        
        # Se métricas não foram fornecidas, computar agora
        if metrics is None:
            metrics = self.compute()
        
        organized_metrics = {}
        
        if self.logger:
            # Log aggregated metrics com timestamp
            organized_metrics.update({
                f"{prefix}mAP": metrics['mAP'],
                f"{prefix}macro_precision": metrics['macro']['precision'],
                f"{prefix}macro_recall": metrics['macro']['recall'],
                f"{prefix}macro_f1": metrics['macro']['f1'],
                f"{prefix}timestamp": timestamp,
                f"{prefix}step": float(step)
            })
            
            # Log per-class metrics com contexto
            for i, cls_name in enumerate(self.class_names):
                organized_metrics.update({
                    f"{prefix}{cls_name}/precision": metrics['per_class']['precision'][i],
                    f"{prefix}{cls_name}/recall": metrics['per_class']['recall'][i],
                    f"{prefix}{cls_name}/f1": metrics['per_class']['f1'][i],
                    f"{prefix}{cls_name}/tp": metrics['per_class']['tp'][i],
                    f"{prefix}{cls_name}/fp": metrics['per_class']['fp'][i],
                    f"{prefix}{cls_name}/fn": metrics['per_class']['fn'][i],
                    f"{prefix}{cls_name}/tn": metrics['per_class']['tn'][i]
                })

            # Log estruturado com contexto
            for key, value in organized_metrics.items():
                self.logger.log(
                    key,
                    value
                )
                
            # Log resumo para debug
            print(f"[{timestamp}] {prefix} mAP: {metrics['mAP']:.4f}, "
                  f"macro_f1: {metrics['macro']['f1']:.4f}, step: {step}")
        else:
            print(f"[{timestamp}] No logger provided, skipping logging for {prefix}")
            
        # Reset should be called manually by the user at the end of an epoch
        self.reset()
















