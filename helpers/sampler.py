import numpy as np
from torch.utils.data import WeightedRandomSampler

class MultiLabelBalancedSampler(WeightedRandomSampler):
    def __init__(self, labels, max_samples=None, num_classes=None, epoch=0):
        if num_classes is None:
            raise ValueError("num_classes é obrigatório para MultiLabelBalancedSampler")
        # Aplicar seed determinístico
        np.random.seed(42 + epoch)
        
        # Calcular frequência de cada classe
        class_counts = np.zeros(num_classes)
        for label in labels:
            class_counts += label
        
        # Calcular pesos para classes (inverso da frequência)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights /= class_weights.sum()
        
        # Calcular peso para cada amostra
        sample_weights = []
        for idx in range(len(labels)):
            label = labels[idx]
            active_classes = np.where(label)[0]
            if len(active_classes) > 0:
                weight = class_weights[active_classes].sum()
            else:
                weight = class_weights.mean()
            sample_weights.append(weight)
        
        # Configurar sampler
        num_samples = max_samples if max_samples else len(labels)
        super().__init__(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True
        )