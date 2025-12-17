"""
Episode sampling for meta-learning.

Implements N-way K-shot episode creation for episodic training and evaluation.
"""

import random
import torch
from typing import Dict, List, Tuple
from pathlib import Path

from src.data.augmentation import load_image


def sample_episode(
    class_images_dict: Dict[str, List[Path]],
    n_way: int,
    k_shot: int,
    q_query: int,
    is_training: bool = True,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a single N-way K-shot episode.
    
    An episode consists of:
    - Support set: N classes × K examples per class
    - Query set: N classes × Q examples per class
    
    Args:
        class_images_dict: Dictionary mapping class names to image paths
        n_way: Number of classes in the episode
        k_shot: Number of support examples per class
        q_query: Number of query examples per class
        is_training: Whether to apply training augmentation
        device: Device to place tensors on ('cuda' or 'cpu')
        
    Returns:
        support_x: Support images [N*K, 1, 224, 224]
        support_y: Support labels [N*K] (values 0 to N-1)
        query_x: Query images [N*Q, 1, 224, 224]
        query_y: Query labels [N*Q] (values 0 to N-1)
        
    Raises:
        ValueError: If not enough eligible classes or samples
    """
    # Filter classes with enough samples
    min_samples = k_shot + q_query
    eligible = [
        c for c, imgs in class_images_dict.items() 
        if len(imgs) >= min_samples
    ]
    
    if len(eligible) < n_way:
        raise ValueError(
            f"Not enough eligible classes for {n_way}-way episode. "
            f"Found {len(eligible)} classes with >={min_samples} samples."
        )
    
    # Randomly select N classes
    selected_classes = random.sample(eligible, n_way)
    
    support_imgs = []
    support_lbls = []
    query_imgs = []
    query_lbls = []

    # Sample support and query for each class
    for label_idx, class_name in enumerate(selected_classes):
        # Randomly sample K+Q images for this class
        all_imgs = class_images_dict[class_name]
        sampled_imgs = random.sample(all_imgs, k_shot + q_query)
        
        # Split into support and query
        support_paths = sampled_imgs[:k_shot]
        query_paths = sampled_imgs[k_shot:]

        # Load support images
        for path in support_paths:
            support_imgs.append(load_image(path, is_training=is_training))
            support_lbls.append(label_idx)
        
        # Load query images
        for path in query_paths:
            query_imgs.append(load_image(path, is_training=is_training))
            query_lbls.append(label_idx)

    # Stack into tensors
    support_x = torch.stack(support_imgs).to(device)
    support_y = torch.tensor(support_lbls, dtype=torch.long, device=device)
    query_x = torch.stack(query_imgs).to(device)
    query_y = torch.tensor(query_lbls, dtype=torch.long, device=device)
    
    return support_x, support_y, query_x, query_y


def sample_batch_episodes(
    class_images_dict: Dict[str, List[Path]],
    n_way: int,
    k_shot: int,
    q_query: int,
    batch_size: int,
    is_training: bool = True,
    device: str = "cuda"
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Sample a batch of episodes.
    
    Args:
        class_images_dict: Dictionary mapping class names to image paths
        n_way: Number of classes per episode
        k_shot: Number of support examples per class
        q_query: Number of query examples per class
        batch_size: Number of episodes to sample
        is_training: Whether to apply training augmentation
        device: Device to place tensors on
        
    Returns:
        List of (support_x, support_y, query_x, query_y) tuples
    """
    episodes = []
    for _ in range(batch_size):
        episode = sample_episode(
            class_images_dict, n_way, k_shot, q_query, is_training, device
        )