import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import random
import math


class MaskAwareTransforms:
    """
    Class that applies the same augmentations as the original transform_train,
    but also generates masks indicating valid regions vs artificial background.
    Mask-aware transforms for WSSS.
    """
    
    def __init__(self, image_size=448, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
    def __call__(self, image, mask=None):
        """
        Args:
            image: PIL Image
            mask: numpy array (H, W) - initial mask (None = all valid)

        Returns:
            transformed_image: torch.Tensor
            final_mask: torch.Tensor (H, W) - mask after transforms
        """
        if mask is None:
            mask = np.ones((image.size[1], image.size[0]), dtype=np.float32)
        
        # Apply sequence of transforms that also update mask
        image, mask = self._apply_randaugment(image, mask)
        image, mask = self._apply_horizontal_flip(image, mask)
        image, mask = self._apply_vertical_flip(image, mask)
        image, mask = self._apply_rotation(image, mask)
        image, mask = self._apply_random_resized_crop(image, mask)
        
        # Ensure image has the correct size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Resize mask to the final size
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask_pil) / 255.0
        
        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).float()
        
        return image, mask
    
    def _apply_randaugment(self, image, mask):
        """Applies RandAugment while keeping the mask."""
        # RandAugment does not affect the mask because it does not create artificial background
        # It only applies transformations that preserve image structure
        # aug = transforms.RandAugment(num_ops=2, magnitude=10)
        # image = aug(image)
        return image, mask
    
    def _apply_horizontal_flip(self, image, mask):
        """Applies horizontal flip with probability 0.5."""
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = np.fliplr(mask)
        return image, mask
    
    def _apply_vertical_flip(self, image, mask):
        """Applies vertical flip with probability 0.5."""
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = np.flipud(mask)
        return image, mask
    
    def _apply_rotation(self, image, mask):
        """Applies random rotation from 0 to 360 degrees with some probability."""
        # Apply rotation with probability (to have more images without artificial background)
        if random.random() < 0.3:  # 30% chance to apply rotation
            angle = random.uniform(0, 360)
            
            # Rotate image WITHOUT padding (uses original border color)
            # This avoids creating artificial background that can confuse the model
            image = image.rotate(angle, expand=False)
            
            # Rotate mask - keep everything valid (no artificial background)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.rotate(angle, expand=False)  # Sem fillcolor, mantém valores originais
            mask = np.array(mask_pil) / 255.0
        
        return image, mask
    
    def _apply_random_resized_crop(self, image, mask):
        """Applies RandomResizedCrop with scale (0.8, 1.2)."""

        if random.random() < 0.5:
            return image, mask



        scale = random.uniform(0.8, 1.2)
        ratio = random.uniform(3/4, 4/3)
        
        w, h = image.size
        
        # Compute new area
        area = w * h * scale
        new_w = int(math.sqrt(area * ratio))
        new_h = int(math.sqrt(area / ratio))
        
        # Ensure it does not exceed image bounds
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        
        # Compute crop position
        if new_w < w:
            left = random.randint(0, w - new_w)
        else:
            left = 0
            
        if new_h < h:
            top = random.randint(0, h - new_h)
        else:
            top = 0
            
        right = left + new_w
        bottom = top + new_h
        
        # Apply crop to image
        image = image.crop((left, top, right, bottom))
        
        # Apply crop to mask
        mask = mask[top:bottom, left:right]
        
        # Resize to final size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Resize mask
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask_pil) / 255.0
        
        return image, mask


class DomainCropAugmentation:
    """
    Domain augmentation that exploits the spatial structure of images.
    Groups classes by region (top/bottom), selects a region with
    configurable probabilities, and applies crop or obscuring.
    Ported from SSDB-II/VSS_dataset.py.
    """

    # Regions defined as fractions (x1, y1, x2, y2) of the image size
    REGION_BOXES = {
        "cima":   (0.25, 0.20, 0.75, 0.40),
        "cima2":  (0.00, 0.00, 1.00, 0.40),
        "baixo":  (0.25, 0.60, 0.75, 0.80),
        "baixo2": (0.00, 0.60, 1.00, 1.00),
        "centro": (0.25, 0.20, 0.75, 0.80),
    }

    def __init__(self, config: dict, image_size: int = 448):
        self.image_size = image_size
        self.enabled = config.get("enabled", False)
        self.min_fg = config.get("min_foreground_classes", 4)
        self.min_fg_centro = config.get("min_foreground_classes_centro", 7)
        self.crop_vs_obscure = config.get("crop_vs_obscure", [0.7, 0.3])
        self.revert_to_normal_prob = config.get("revert_to_normal_prob", 0.7)

        rp = config.get("region_probabilities", {
            "normal": 0.10, "cima": 0.30, "baixo": 0.15,
            "centro": 0.15, "cima2": 0.15, "baixo2": 0.15,
        })
        self.region_names = list(rp.keys())
        self.region_probs = [rp[k] for k in self.region_names]

        sg = config.get("spatial_groups", {
            "top": [0, 2, 6, 7, 9, 12, 15, 10],
            "bottom": [3, 4, 5, 8, 10, 11, 13, 14],
        })
        self.classes_top = sg["top"]
        self.classes_bottom = sg["bottom"]

    def __call__(self, image, label_fg, mask):
        """
        Args:
            image: PIL.Image RGB
            label_fg: np.ndarray (num_foreground,) float32, 0-based foreground
            mask: np.ndarray (H, W) float32
        Returns:
            (image, label_fg, mask) — possibly modified
        """
        if not self.enabled:
            return image, label_fg, mask

        w, h = image.size
        new_label = label_fg.copy()
        num_fg = int(new_label.sum())

        # Select region
        region = np.random.choice(self.region_names, p=self.region_probs)

        # Override based on present classes
        region, pao_case = self._apply_overrides(region, label_fg)

        # Probability of reverting to normal
        if region != "normal" and np.random.random() < self.revert_to_normal_prob:
            region = "normal"

        # Check minimum number of foreground classes
        if region == "normal" or num_fg < self.min_fg or \
                (region == "centro" and num_fg < self.min_fg_centro):
            return image, label_fg, mask

        # Select operation
        op = np.random.choice(["crop", "ofuscar"], p=self.crop_vs_obscure)

        # Apply zeroing of incompatible classes
        new_label = self._zero_incompatible(region, new_label, pao_case)

        # Safety: if all classes became zero, revert
        if new_label.sum() == 0:
            return image, label_fg, mask

        # Compute box in pixels
        box = self._get_box_pixels(region, w, h)

        # Aplicar operação
        if op == "crop":
            image, mask = self._apply_crop(image, mask, box)
        else:
            image, mask = self._apply_obscure(image, mask, box, w, h)

        return image, new_label, mask

    def _apply_overrides(self, region, label_fg):
        """Region override based on present classes."""
        pao_case = False

        # If no top classes are present → force top
        top_check = [i for i in self.classes_top if i < len(label_fg)]
        if top_check and sum(label_fg[i] for i in top_check) == 0:
            region = "cima" if np.random.random() < 0.5 else "cima2"
            pao_case = True

        return region, pao_case

    def _zero_incompatible(self, region, label, pao_case):
        """Zeros out classes incompatible with the selected region."""
        if region in ("cima", "cima2"):
            for idx in self.classes_bottom:
                if idx < len(label):
                    label[idx] = 0
        elif region in ("baixo", "baixo2"):
            for idx in self.classes_top:
                if idx < len(label):
                    label[idx] = 0
        elif region == "centro":
            # Center: zero out classes at the extremes (top and bottom border)
            # Classes that lie exclusively on the borders are zeroed
            pass
        return label

    def _get_box_pixels(self, region, w, h):
        """Converts fractional region to pixel coordinates."""
        if region == "normal":
            return (0, 0, w, h)
        fx1, fy1, fx2, fy2 = self.REGION_BOXES[region]
        return (int(fx1 * w), int(fy1 * h), int(fx2 * w), int(fy2 * h))

    def _apply_crop(self, image, mask, box):
        """Direct crop of the selected region."""
        x1, y1, x2, y2 = box
        image = image.crop(box)
        mask = mask[y1:y2, x1:x2]
        return image, mask

    def _apply_obscure(self, image, mask, box, w, h):
        """Obscure: gray background + paste of the region at a random position."""
        gray_bg = Image.new("RGB", (w, h), (128, 128, 128))
        cropped = image.crop(box)

        # Random scale
        sx = np.random.uniform(0.8, 1.2)
        sy = np.random.uniform(0.8, 1.2)
        cw, ch = cropped.size
        new_w = int(cw * sx)
        new_h = int(ch * sy)
        scaled = cropped.resize((new_w, new_h))

        # Random position
        max_x = max(0, w - new_w)
        max_y = max(0, h - new_h)

        if max_x > 0 and max_y > 0:
            rx = np.random.randint(0, max_x + 1)
            ry = np.random.randint(0, max_y + 1)
        else:
            rx, ry = 0, 0
            if new_w > w or new_h > h:
                sf = min(w / new_w, h / new_h)
                new_w = int(new_w * sf)
                new_h = int(new_h * sf)
                scaled = scaled.resize((new_w, new_h))

        gray_bg.paste(scaled, (rx, ry))
        image = gray_bg

        # Mask: only the pasted region is valid
        mask = np.zeros((h, w), dtype=np.float32)
        mask[ry:ry + new_h, rx:rx + new_w] = 1.0

        return image, mask


class MaskAwareValidationTransforms:
    """
    Validation transforms - only resize and normalization.
    """
    
    def __init__(self, image_size=448, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.image_size = image_size
        self.mean = mean
        self.std = std
    
    def __call__(self, image, mask=None):
        """
        Args:
            image: PIL Image
            mask: numpy array (H, W) - initial mask (None = all valid)

        Returns:
            transformed_image: torch.Tensor
            final_mask: torch.Tensor (H, W) - mask after transforms
        """
        if mask is None:
            mask = np.ones((image.size[1], image.size[0]), dtype=np.float32)
        
        # Only resize and normalization
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Resize mask
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask_pil) / 255.0
        
        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).float()
        
        return image, mask







