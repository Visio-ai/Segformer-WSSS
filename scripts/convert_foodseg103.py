"""
Converte o dataset FoodSeg103 (HuggingFace) para o formato local definido em plans/dataset.md.

Uso:
    python scripts/convert_foodseg103.py --output-dir /path/to/FoodSeg103_converted

Requisitos: pip install datasets huggingface_hub pyyaml opencv-python pillow numpy
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError:
    yaml = None

# Nomes das classes FoodSeg103 (id 0 = background, 1-103 = ingredientes)
FOODSEG103_CLASS_NAMES = {
    0: "background",
    1: "candy",
    2: "egg tart",
    3: "french fries",
    4: "chocolate",
    5: "biscuit",
    6: "popcorn",
    7: "pudding",
    8: "ice cream",
    9: "cheese butter",
    10: "cake",
    11: "wine",
    12: "milkshake",
    13: "coffee",
    14: "juice",
    15: "milk",
    16: "tea",
    17: "almond",
    18: "red beans",
    19: "cashew",
    20: "dried cranberries",
    21: "soy",
    22: "walnut",
    23: "peanut",
    24: "egg",
    25: "apple",
    26: "date",
    27: "apricot",
    28: "avocado",
    29: "banana",
    30: "strawberry",
    31: "cherry",
    32: "blueberry",
    33: "raspberry",
    34: "mango",
    35: "olives",
    36: "peach",
    37: "lemon",
    38: "pear",
    39: "fig",
    40: "pineapple",
    41: "grape",
    42: "kiwi",
    43: "melon",
    44: "orange",
    45: "watermelon",
    46: "steak",
    47: "pork",
    48: "chicken duck",
    49: "sausage",
    50: "fried meat",
    51: "lamb",
    52: "sauce",
    53: "crab",
    54: "fish",
    55: "shellfish",
    56: "shrimp",
    57: "soup",
    58: "bread",
    59: "corn",
    60: "hamburg",
    61: "pizza",
    62: "hanamaki baozi",
    63: "wonton dumplings",
    64: "pasta",
    65: "noodles",
    66: "rice",
    67: "pie",
    68: "tofu",
    69: "eggplant",
    70: "potato",
    71: "garlic",
    72: "cauliflower",
    73: "tomato",
    74: "kelp",
    75: "seaweed",
    76: "spring onion",
    77: "rape",
    78: "ginger",
    79: "okra",
    80: "lettuce",
    81: "pumpkin",
    82: "cucumber",
    83: "white radish",
    84: "carrot",
    85: "asparagus",
    86: "bamboo shoots",
    87: "broccoli",
    88: "celery stick",
    89: "cilantro mint",
    90: "snow peas",
    91: "cabbage",
    92: "bean sprouts",
    93: "onion",
    94: "pepper",
    95: "green beans",
    96: "French beans",
    97: "king oyster mushroom",
    98: "shiitake",
    99: "enoki mushroom",
    100: "oyster mushroom",
    101: "white button mushroom",
    102: "salad",
    103: "other ingredients",
}

NUM_CLASSES = 104
NUM_FOREGROUND = 103


def mask_to_yolo_polygons(mask_arr, height, width, class_ids=None):
    """
    Converte máscara pixel (H, W) com valores 0..103 em linhas YOLO polygon.
    Retorna lista de strings "class_id x1 y1 x2 y2 ..." (coordenadas normalizadas).
    Desconsidera background (0). class_ids opcional: set de classes a exportar.
    """
    if cv2 is None:
        raise ImportError("opencv-python é necessário. pip install opencv-python")
    if class_ids is None:
        class_ids = set(range(1, NUM_CLASSES))
    lines = []
    for c in range(1, NUM_CLASSES):
        if c not in class_ids:
            continue
        binary = (mask_arr == c).astype(np.uint8)
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) < 2:
                continue
            cnt = cnt.reshape(-1, 2)
            # Normalizar para [0, 1]
            xs = (cnt[:, 0] / width).tolist()
            ys = (cnt[:, 1] / height).tolist()
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
            lines.append(f"{c} {coords}")
    return lines


def multihot_from_mask(mask_arr, num_foreground=NUM_FOREGROUND):
    """Deriva vetor multi-hot (sem background) a partir da máscara. 1 = classe presente."""
    present = set(np.unique(mask_arr).tolist()) - {0}
    vec = [1 if (i + 1) in present else 0 for i in range(num_foreground)]
    return vec


def main():
    parser = argparse.ArgumentParser(
        description="Converte FoodSeg103 (HuggingFace) para formato local (plans/dataset.md)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Diretório raiz onde criar o dataset convertido",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="EduardoPacheco/FoodSeg103",
        help="Nome do dataset no HuggingFace",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Diretório de cache do HuggingFace datasets (opcional)",
    )
    args = parser.parse_args()

    if yaml is None:
        print("Erro: PyYAML necessário. pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    if cv2 is None:
        print("Erro: opencv-python necessário. pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Erro: datasets necessário. pip install datasets", file=sys.stderr)
        sys.exit(1)

    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)
    train_img_dir = os.path.join(out_root, "train", "images")
    val_img_dir = os.path.join(out_root, "val", "images")
    val_labels_dir = os.path.join(out_root, "val", "labels")
    annotations_dir = os.path.join(out_root, "annotations")
    splits_single = os.path.join(out_root, "splits", "single")
    for d in (train_img_dir, val_img_dir, val_labels_dir, annotations_dir, splits_single):
        os.makedirs(d, exist_ok=True)

    print("Carregando dataset do HuggingFace...")
    kwargs = {}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    ds = load_dataset(args.dataset_name, **kwargs)

    # Inferir nome da coluna de máscara
    train_split = ds.get("train", ds.get("training", None))
    if train_split is None:
        train_split = list(ds.values())[0]
    ex = train_split[0]
    mask_key = None
    for k in ("semantic_segmentation", "annotation", "label", "pixel_label", "mask", "segmentation"):
        if k in ex and ex[k] is not None:
            mask_key = k
            break
    if mask_key is None:
        print("Erro: não foi possível encontrar coluna de máscara no dataset.", file=sys.stderr)
        print("Chaves disponíveis:", list(ex.keys()), file=sys.stderr)
        sys.exit(1)
    image_key = "image" if "image" in ex else "img"
    if image_key not in ex:
        print("Erro: coluna 'image' não encontrada. Chaves:", list(ex.keys()), file=sys.stderr)
        sys.exit(1)

    train_labels = {}
    train_names = []
    val_labels = {}
    val_names = []
    stats_train = 0
    stats_val = 0

    # Train
    print("Processando split de treino...")
    for idx, row in enumerate(train_split):
        img = row[image_key]
        if hasattr(img, "save"):
            pil = img
        else:
            pil = Image.fromarray(np.array(img))
        name_base = f"train_{idx:05d}"
        train_names.append(name_base)
        out_path = os.path.join(train_img_dir, f"{name_base}.jpg")
        pil.save(out_path, "JPEG", quality=95)
        mask = row[mask_key]
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask.squeeze()
        train_labels[name_base] = multihot_from_mask(mask)
        stats_train += 1
        if (idx + 1) % 500 == 0:
            print(f"  Train: {idx + 1} imagens")

    # Validation
    val_split = ds.get("validation", ds.get("val", None))
    if val_split is None:
        val_split = list(ds.values())[1] if len(ds) > 1 else []
    print("Processando split de validação...")
    for idx, row in enumerate(val_split):
        img = row[image_key]
        if hasattr(img, "save"):
            pil = img
        else:
            pil = Image.fromarray(np.array(img))
        name_base = f"val_{idx:05d}"
        val_names.append(name_base)
        out_path = os.path.join(val_img_dir, f"{name_base}.jpg")
        pil.save(out_path, "JPEG", quality=95)
        mask = row[mask_key]
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask.squeeze()
        h, w = mask.shape[0], mask.shape[1]
        val_labels[name_base] = multihot_from_mask(mask)
        txt_path = os.path.join(val_labels_dir, f"{name_base}.txt")
        lines = mask_to_yolo_polygons(mask, h, w)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        # Cache PNG: pixel value = class_id (0..103)
        png_path = os.path.join(val_labels_dir, f"{name_base}.png")
        out_mask = np.zeros((h, w), dtype=np.uint8)
        for c in range(NUM_CLASSES):
            out_mask[mask == c] = c
        Image.fromarray(out_mask).save(png_path)
        stats_val += 1
        if (idx + 1) % 500 == 0:
            print(f"  Val: {idx + 1} imagens")

    with open(os.path.join(annotations_dir, "train_labels.json"), "w") as f:
        json.dump(train_labels, f, indent=0)
    with open(os.path.join(annotations_dir, "val_labels.json"), "w") as f:
        json.dump(val_labels, f, indent=0)

    with open(os.path.join(splits_single, "train.txt"), "w") as f:
        f.write("\n".join(train_names) + "\n")
    with open(os.path.join(splits_single, "val.txt"), "w") as f:
        f.write("\n".join(val_names) + "\n")

    class_names_yaml = {str(k): v for k, v in FOODSEG103_CLASS_NAMES.items()}
    dataset_yaml = {
        "name": "FoodSeg103",
        "version": "1.0",
        "description": "FoodSeg103 convertido do HuggingFace para formato local (plans/dataset.md)",
        "task": "weakly_supervised_semantic_segmentation",
        "supervision": {"train": "image_level", "val": "pixel_level"},
        "num_classes": NUM_CLASSES,
        "num_foreground_classes": NUM_FOREGROUND,
        "class_names": class_names_yaml,
        "label_format": {
            "classification": {"type": "multi_hot", "length": NUM_FOREGROUND, "file_format": "json"},
            "segmentation": {
                "type": "yolo_polygon",
                "file_format": "txt",
                "mask_cache": "png",
                "num_classes_with_bg": NUM_CLASSES,
            },
        },
        "image_format": "jpg",
        "image_channels": 3,
        "cross_validation": {"supported_modes": ["single", "kfold"], "default_n_folds": 4, "seed": 42, "strategy": "random"},
        "stats": {"train_images": stats_train, "val_images": stats_val, "total_images": stats_train + stats_val},
    }
    with open(os.path.join(out_root, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    stats = {
        "mode": "single",
        "train_count": stats_train,
        "val_count": stats_val,
    }
    with open(os.path.join(splits_single, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Conversão concluída: {out_root}")
    print(f"  Train: {stats_train} imagens, labels em annotations/train_labels.json")
    print(f"  Val: {stats_val} imagens, labels YOLO em val/labels/*.txt, annotations/val_labels.json")
    print(f"  dataset.yaml e splits/single/ gerados.")


if __name__ == "__main__":
    main()
