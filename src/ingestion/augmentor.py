from __future__ import annotations
from typing import Any, Optional
try:
    from PIL import Image, ImageEnhance
except ImportError:
    Image = None
    ImageEnhance = None
import random

try:
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class MetrologyAugmentor:
    """
    Data Augmentation pipeline specifically tailored for optical microscopy and SEM die micrographs.
    Applies physical invariant transformations (orthogonal rotations, axial reflections)
    and cleanroom illumination variations to boost sample efficiency in few-shot regimes.
    """
    def __init__(self, target_size: int = 518):
        self.target_size = target_size

    def get_torch_train_transform(self) -> Any:
        """Returns standard TorchVision augmentation pipeline for few-shot training."""
        if not TORCHVISION_AVAILABLE:
            return None

        return T.Compose([
            T.Resize((self.target_size, self.target_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=(-180, 180)),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_torch_eval_transform(self) -> Any:
        """Returns deterministic TorchVision preprocessing pipeline for validation & test inference."""
        if not TORCHVISION_AVAILABLE:
            return None

        return T.Compose([
            T.Resize((self.target_size, self.target_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def augment_pil_image(self, img: Any) -> Any:
        """Applies random cleanroom optical transforms directly on PIL image without Torch dependency."""
        out = img.copy()

        # Random rotation (90, 180, 270)
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            out = out.rotate(angle)

        # Random horizontal flip
        if random.random() > 0.5:
            out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5:
            out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Slight contrast jitter
        factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Contrast(out)
        out = enhancer.enhance(factor)

        return out
