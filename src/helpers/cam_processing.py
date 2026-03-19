import numpy as np


def balance_class_presence(cam, multi_hot_label, min_fraction=0.03, factor=1.05, max_iter=100):
    """
    Balanceia a presença de classes na CAM garantindo que cada classe presente
    tenha pelo menos min_fraction (3%) dos pixels no mapa de segmentação final.

    Args:
        cam: np.ndarray de shape [C, D, D] com a Class Activation Map
        multi_hot_label: np.ndarray de shape [C] com 0s e 1s indicando classes presentes
        min_fraction: Fração mínima de pixels requerida (default=0.03)
        factor: Fator multiplicativo para ajuste de pesos (default=1.05)
        max_iter: Número máximo de iterações (default=100)

    Returns:
        weighted_cam: CAM ponderada [C, D, D]
        final_weights: Pesos finais aplicados [C]
    """
    C, D, _ = cam.shape
    weights = np.ones(C)
    total_pixels = D * D

    for _ in range(max_iter):
        # Aplica os pesos aos canais da CAM
        weighted_cam = cam * weights[:, np.newaxis, np.newaxis]

        # Calcula o mapa de segmentação
        seg_map = weighted_cam.argmax(axis=0)

        # Calcula a fração de pixels por classe
        class_counts = np.bincount(seg_map.ravel(), minlength=C)
        class_fractions = class_counts / total_pixels

        # Identifica classes presentes abaixo do limiar
        below_threshold = np.where(
            (multi_hot_label == 1) &
            (class_fractions < min_fraction)
        )[0]

        # Critério de parada: todas as classes satisfazem o mínimo
        if len(below_threshold) == 0:
            break

        # Aumenta pesos das classes insuficientes
        weights[below_threshold] *= factor

    # Retorna a CAM final e os pesos
    final_weighted_cam = cam * weights[:, np.newaxis, np.newaxis]
    return final_weighted_cam, weights
