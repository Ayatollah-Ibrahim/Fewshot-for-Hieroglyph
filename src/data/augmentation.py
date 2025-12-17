"""
Data augmentation strategies for hieroglyphic images.

Provides training and validation transforms with appropriate augmentations
to improve model generalization while preserving hieroglyphic structure.
"""

from PIL import Image
from torchvision import transforms
from typing import Callable


def get_train_transform() -> Callable:
    """
    Get training augmentation pipeline.
    
    Applies aggressive augmentation including:
    - Random rotation (±15°)
    - Random affine transformations
    - Random perspective distortion
    - Color jittering
    - Random erasing (simulates occlusions)
    
    Returns:
        Composed torchvision transform for training
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        
        # Geometric augmentations
        transforms.RandomRotation(15),  # Hieroglyphs can be at angles
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Small translations
            scale=(0.9, 1.1)        # Small scaling
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Appearance augmentations
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        
        # Normalization
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        
        # Random erasing (simulates occlusions/damage)
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])


def get_val_transform() -> Callable:
    """
    Get validation/test transformation pipeline.
    
    No augmentation - only basic preprocessing:
    - Resize
    - Grayscale conversion
    - Normalization
    
    Returns:
        Composed torchvision transform for validation/testing
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def load_image(path: str, is_training: bool = True):
    """
    Load and transform an image.
    
    Args:
        path: Path to image file
        is_training: If True, apply training augmentation; 
                    otherwise apply validation transform
        
    Returns:
        Transformed image tensor [1, H, W]
    """
    img = Image.open(path).convert("L")  # Convert to grayscale
    transform = get_train_transform() if is_training else get_val_transform()
    return transform(img)


