import os
import math
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.models.base import DefectClassifierInterface

DIE_DEFECT_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]


def build_linear_probe(embed_dim: int, num_classes: int) -> nn.Sequential:
    """Builds the classification head shared by training and inference."""
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, num_classes),
    )


class DieVFMModel(nn.Module):
    """
    Vision Foundation Model (DINOv2) with Few-Shot Classification Head.
    Set unfreeze_blocks > 0 to jointly fine-tune the final transformer blocks
    instead of using DINOv2 only as a frozen feature extractor.
    """
    def __init__(self, backbone_name: str = "dinov2_vitb14", num_classes: int = 6, unfreeze_blocks: int = 0):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.unfreeze_blocks = unfreeze_blocks

        # Load Vision Transformer
        try:
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        except Exception:
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, source="local")

        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze_blocks > 0:
            for param in self.backbone.blocks[-unfreeze_blocks:].parameters():
                param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        self.embed_dim = getattr(self.backbone, "embed_dim", 768)

        self.head = build_linear_probe(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


class DieVFMClassifier(DefectClassifierInterface):
    """
    Vision Foundation Model (VFM) Classifier for Semiconductor & PCB Micro-Metrology.
    Uses NV-DINOv2 / DINOv2 self-supervised visual representations for sub-micron physical line defect classification.
    """
    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        num_classes: int = 6,
        embedding_dim: Optional[int] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        unfreeze_blocks: int = 0
    ):
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.classes = DIE_DEFECT_CLASSES
        self.custom_embedding_dim = embedding_dim

        # Auto-detect compute hardware (Apple Silicon MPS, NVIDIA CUDA, or CPU)
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Invalid VFM checkpoint: {weights_path}")
            backbone_name = checkpoint.get("backbone", checkpoint.get("backbone_name", backbone_name))
            checkpoint_classes = checkpoint.get("classes", DIE_DEFECT_CLASSES)
            if checkpoint_classes != DIE_DEFECT_CLASSES:
                raise ValueError("Checkpoint defect classes do not match the deployed classifier")

        self.backbone_name = backbone_name
        unfreeze_blocks = checkpoint.get("unfreeze_blocks", unfreeze_blocks) if weights_path and os.path.exists(weights_path) else unfreeze_blocks
        self.model = DieVFMModel(backbone_name=backbone_name, num_classes=num_classes, unfreeze_blocks=unfreeze_blocks).to(self.device)
        self.torch_model = self.model
        self.torch_head = self.model.head

        # Standard cleanroom evaluation transform
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.use_pytorch = True
        self.weights = []
        self.bias = []

        if weights_path:
            self.load_checkpoint(weights_path)
        else:
            self._sync_weights_from_head()

    def _sync_weights_from_head(self):
        """Syncs head parameters to CPU python lists for inspection and export."""
        if self.model and self.model.head:
            with torch.no_grad():
                linear_layer = self.model.head[-1]
                self.weights = linear_layer.weight.t().cpu().tolist()
                self.bias = linear_layer.bias.cpu().tolist()

    def extract_features(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> Any:
        """Extracts DINOv2 representations from input micrographs."""
        self.model.eval()
        if isinstance(images, Image.Image):
            tensor = self.eval_transform(images).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.backbone(tensor)[0].cpu()
                if self.custom_embedding_dim and len(feat) != self.custom_embedding_dim:
                    if self.custom_embedding_dim > len(feat):
                        feat = torch.cat([feat, torch.zeros(self.custom_embedding_dim - len(feat))])
                    else:
                        feat = feat[:self.custom_embedding_dim]
                return feat.tolist()
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
            tensors = torch.stack([self.eval_transform(img) for img in images]).to(self.device)
            with torch.no_grad():
                return self.model.backbone(tensors)
        elif isinstance(images, torch.Tensor):
            tensor = images.to(self.device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                return self.model.backbone(tensor)
        else:
            raise ValueError("Unsupported input format for extract_features")

    def predict_logits(self, image_data: Any) -> List[float]:
        """Runs forward pass through DINOv2 and returns class logits."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(image_data, Image.Image):
                tensor = self.eval_transform(image_data).unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                return logits[0].cpu().tolist()
            elif isinstance(image_data, torch.Tensor):
                tensor = image_data.to(self.device)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                logits = self.model(tensor)
                return logits[0].cpu().tolist()
            else:
                return [0.0] * self.num_classes

    def softmax(self, logits: List[float]) -> List[float]:
        """Computes numerically stable softmax probabilities."""
        max_l = max(logits)
        exp_l = [math.exp(x - max_l) for x in logits]
        sum_exp = sum(exp_l) + 1e-8
        return [x / sum_exp for x in exp_l]

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: Optional[int] = None,
        val_accuracy: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Saves PyTorch state dict and training metadata."""
        checkpoint_path = str(checkpoint_path)
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        self._sync_weights_from_head()

        torch.save({
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "backbone_name": self.backbone_name,
            "unfreeze_blocks": self.model.unfreeze_blocks,
            "model_state_dict": self.model.state_dict(),
            "head_state_dict": self.model.head.state_dict(),
            "metadata": metadata or {}
        }, checkpoint_path)
        return checkpoint_path

    def save_safetensors(self, safetensors_path: Union[str, Path]) -> str:
        """Saves linear probe head in Hugging Face SafeTensors format."""
        safetensors_path = str(safetensors_path)
        os.makedirs(os.path.dirname(os.path.abspath(safetensors_path)), exist_ok=True)
        try:
            from safetensors.torch import save_file
            tensors_dict = {
                f"head_{k}": v.contiguous() for k, v in self.model.head.state_dict().items()
            }
            save_file(tensors_dict, safetensors_path)
            return safetensors_path
        except Exception:
            return safetensors_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads weights from disk."""
        checkpoint_path = str(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            return {}

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid VFM checkpoint: {checkpoint_path}")
        if ckpt.get("backbone", ckpt.get("backbone_name", self.backbone_name)) != self.backbone_name:
            raise ValueError("Checkpoint backbone does not match the initialized classifier")
        if ckpt.get("classes", DIE_DEFECT_CLASSES) != self.classes:
            raise ValueError("Checkpoint defect classes do not match the initialized classifier")
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        elif "head_state_dict" in ckpt:
            self.model.head.load_state_dict(ckpt["head_state_dict"], strict=True)
        else:
            raise ValueError("Checkpoint is missing model_state_dict or head_state_dict")
        self.model.eval()
        self._sync_weights_from_head()
        res = {"epoch": ckpt.get("epoch"), "val_accuracy": ckpt.get("val_accuracy")}
        if "metadata" in ckpt and isinstance(ckpt["metadata"], dict):
            res.update(ckpt["metadata"])
        return res

    def classify_patch(self, image_data: Any) -> Dict[str, Any]:
        """Classifies an optical die crop and returns structured probabilities."""
        logits = self.predict_logits(image_data)
        probs = self.softmax(logits)
        pred_idx = probs.index(max(probs))
        confidence = float(probs[pred_idx])
        return {
            "predicted_class": self.classes[pred_idx % len(self.classes)],
            "class_index": pred_idx,
            "confidence": round(confidence, 4),
            "all_probabilities": {cls: round(float(p), 4) for cls, p in zip(self.classes, probs)}
        }
