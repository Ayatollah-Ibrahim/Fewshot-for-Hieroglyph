"""
Data preprocessing utilities for hieroglyphic images.

Handles image loading, contrast enhancement, and dataset organization.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def preprocess_image(img_path: str, img_size: int = 224) -> np.ndarray:
    """
    Preprocess a single hieroglyphic image.
    
    Applies:
    - Grayscale conversion
    - CLAHE contrast enhancement
    - Background flattening
    - Normalization
    - Resizing
    
    Args:
        img_path: Path to input image
        img_size: Target size for resizing
        
    Returns:
        Preprocessed image array [H, W]
        
    Raises:
        ValueError: If image cannot be read
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast normalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Illumination correction (background flattening)
    blur = cv2.GaussianBlur(gray, (55, 55), 0)
    corrected = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Normalize to [0, 255]
    normalized = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    # Resize to target size
    resized = cv2.resize(normalized, (img_size, img_size))

    return resized


class ClassFolderDataset:
    """
    Dataset loader for class-organized folder structure.
    
    Expected structure:
        root/
        ├── class_1/
        │   ├── img1.png
        │   ├── img2.png
        │   └── ...
        ├── class_2/
        │   └── ...
        └── ...
    
    Attributes:
        root: Root directory path
        classes: Sorted list of class names
        class_to_idx: Mapping from class name to index
        class_images: Dict mapping class names to list of image paths
    """
    
    def __init__(self, root: str):
        """
        Initialize dataset from folder structure.
        
        Args:
            root: Path to root directory containing class folders
            
        Raises:
            AssertionError: If root path doesn't exist
        """
        self.root = Path(root)
        assert self.root.exists(), f"Path not found: {root}"
        
        # Get sorted list of class directories
        self.classes = sorted([
            d.name for d in self.root.iterdir() 
            if d.is_dir()
        ])
        
        # Create class to index mapping
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Collect image paths for each class
        self.class_images = {}
        for c in self.classes:
            imgs = sorted(list((self.root / c).glob("*.*")))
            if len(imgs) == 0:
                continue
            self.class_images[c] = imgs

    def num_classes(self) -> int:
        """Return number of classes with images."""
        return len(self.class_images)

    def class_counts(self) -> Dict[str, int]:
        """
        Get sample count for each class.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        return {c: len(imgs) for c, imgs in self.class_images.items()}
    
    def get_class_images(self, class_name: str) -> List[Path]:
        """
        Get all image paths for a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of image paths
            
        Raises:
            KeyError: If class doesn't exist
        """
        return self.class_images[class_name]
    
    def __len__(self) -> int:
        """Total number of images across all classes."""
        return sum(len(imgs) for imgs in self.class_images.values())
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        return (f"ClassFolderDataset(root={self.root}, "
                f"classes={self.num_classes()}, "
                f"total_images={len(self)})")

