from __future__ import annotations
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

DIE_DEFECT_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

class PCBDefectDatasetLoader:
    """
    Ingests and organizes optical microscopy micrographs from the Kaggle PCB Defect dataset.
    Extracts high-resolution localized defect patches using bounding box annotations.
    """
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        k_shot: int = 10,
        val_ratio: float = 0.2,
        target_size: Tuple[int, int] = (518, 518),
        crop_padding: int = 150
    ):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "pcb_dataset"
        self.k_shot = k_shot
        self.val_ratio = val_ratio
        self.target_size = target_size
        self.crop_padding = crop_padding
        self.classes = DIE_DEFECT_CLASSES

    def discover_image_files(self) -> Dict[str, List[Path]]:
        discovered: Dict[str, List[Path]] = {cls: [] for cls in self.classes}
        if not self.data_dir.exists():
            return discovered

        sorted_classes = sorted(self.classes, key=lambda c: len(c), reverse=True)
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".ppm"]:
                parent_name = file_path.parent.name.lower().replace("-", "_").replace(" ", "_")
                stem_name = file_path.stem.lower().replace("-", "_").replace(" ", "_")
                for cls in sorted_classes:
                    if parent_name == cls or parent_name == f"{cls}s" or cls in stem_name or cls in parent_name:
                        discovered[cls].append(file_path)
                        break
        return discovered

    def get_stratified_split(
        self,
        k_shot_train: Optional[int] = None,
        val_ratio: Optional[float] = None,
        seed: int = 42
    ) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, List[Path]]]:
        k_shot = k_shot_train if k_shot_train is not None else self.k_shot
        v_ratio = val_ratio if val_ratio is not None else self.val_ratio
        all_files = self.discover_image_files()
        
        train_split: Dict[str, List[Path]] = {}
        val_split: Dict[str, List[Path]] = {}
        test_split: Dict[str, List[Path]] = {}
        rng = random.Random(seed)

        for cls, paths in all_files.items():
            shuffled = paths[:]
            rng.shuffle(shuffled)
            if len(shuffled) >= k_shot:
                train_split[cls] = shuffled[:k_shot]
                remaining = shuffled[k_shot:]
            else:
                train_split[cls] = shuffled[:]
                remaining = []

            if remaining:
                val_count = max(1, int(len(remaining) * v_ratio)) if len(remaining) > 1 else 0
                val_split[cls] = remaining[:val_count]
                test_split[cls] = remaining[val_count:]
            else:
                val_split[cls] = []
                test_split[cls] = []

        return train_split, val_split, test_split

    def load_datasets(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        train_dict, val_dict, test_dict = self.get_stratified_split()
        
        train_list = []
        for cls, paths in train_dict.items():
            for p in paths:
                train_list.append({"image_path": str(p), "label": self.classes.index(cls), "class_name": cls})
                
        val_list = []
        for cls, paths in val_dict.items():
            for p in paths:
                val_list.append({"image_path": str(p), "label": self.classes.index(cls), "class_name": cls})
                
        test_list = []
        for cls, paths in test_dict.items():
            for p in paths:
                test_list.append({"image_path": str(p), "label": self.classes.index(cls), "class_name": cls})

        return train_list, val_list, test_list

    def get_k_shot_split(self, k_shot: int = 10) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
        train_split, val_split, test_split = self.get_stratified_split(k_shot_train=k_shot, val_ratio=0.0)
        return train_split, test_split

    def find_annotation_xml(self, image_path: Path) -> Optional[Path]:
        """Locates corresponding Pascal VOC XML annotation file for given image."""
        # Standard PCB dataset structure: 
        # Images: PCB_DATASET/images/<Class>/<stem>.jpg
        # Annotations: PCB_DATASET/Annotations/<Class>/<stem>.xml
        
        # Check standard Kaggle hierarchy
        parent_class = image_path.parent.name
        
        # Try path relative to images folder
        if "images" in image_path.parts:
            # find index of 'images'
            img_idx = image_path.parts.index("images")
            # reconstruct path replacing 'images' with 'Annotations'
            parts = list(image_path.parts)
            parts[img_idx] = "Annotations"
            xml_candidate = Path(*parts).with_suffix(".xml")
            if xml_candidate.exists():
                return xml_candidate
                
        # Try sibling directory lookup (common fallback)
        candidates = [
            image_path.parents[1] / "Annotations" / parent_class / f"{image_path.stem}.xml",
            image_path.parents[2] / "Annotations" / parent_class / f"{image_path.stem}.xml",
            image_path.parent / f"{image_path.stem}.xml",
        ]
        
        for c in candidates:
            if c.exists():
                return c
        return None

    def load_and_preprocess_image(self, path: Union[str, Path]) -> Any:
        from PIL import Image
        import xml.etree.ElementTree as ET
        
        path = Path(path)
        img = Image.open(path).convert("RGB")
        xml_path = self.find_annotation_xml(path)
        
        if xml_path and xml_path.exists():
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Check ALL bounding boxes and take the largest one to give the model the most context
                best_box = None
                max_area = 0
                for obj in root.findall("object"):
                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        xmin = int(bndbox.find("xmin").text)
                        ymin = int(bndbox.find("ymin").text)
                        xmax = int(bndbox.find("xmax").text)
                        ymax = int(bndbox.find("ymax").text)
                        area = (xmax - xmin) * (ymax - ymin)
                        if area > max_area:
                            max_area = area
                            best_box = (xmin, ymin, xmax, ymax)
                            
                if best_box:
                    xmin, ymin, xmax, ymax = best_box
                    xmin = max(0, xmin - self.crop_padding)
                    ymin = max(0, ymin - self.crop_padding)
                    xmax = min(img.width, xmax + self.crop_padding)
                    ymax = min(img.height, ymax + self.crop_padding)
                    
                    if xmax > xmin and ymax > ymin:
                        img = img.crop((xmin, ymin, xmax, ymax))
            except Exception as e:
                pass

        # Ensure image is resized to target dimension for VFM
        return img.resize(self.target_size, Image.Resampling.BILINEAR)
