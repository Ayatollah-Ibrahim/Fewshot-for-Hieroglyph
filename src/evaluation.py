"""
Evaluation utilities for meta-learning models.

Implements episode-based evaluation with proper handling of train/val modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

from src.data import sample_episode


def episode_loss_and_acc(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    smoothing: float = 0.0
) -> Tuple[torch.Tensor, float]:
    """
    Compute loss and accuracy for a single episode.
    
    Args:
        model: Model to evaluate
        support_x: Support images [N*K, C, H, W]
        support_y: Support labels [N*K]
        query_x: Query images [Q, C, H, W]
        query_y: Query labels [Q]
        smoothing: Label smoothing factor (0 = no smoothing)
        
    Returns:
        loss: Cross-entropy loss
        acc: Classification accuracy
    """
    logits = model(support_x, support_y, query_x)
    
    if smoothing > 0:
        n_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / (n_classes - 1))
            true_dist.scatter_(1, query_y.unsqueeze(1), 1.0 - smoothing)
        loss = (-true_dist * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(logits, query_y)
    
    pred = logits.argmax(dim=1)
    acc = (pred == query_y).float().mean().item()
    
    return loss, acc


def evaluate_on_episodes(
    model: nn.Module,
    class_images_dict: Dict,
    n_way: int,
    k_shot: int,
    q_query: int,
    episodes: int = 200,
    is_training: bool = False,
    device: str = "cuda"
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate model on multiple episodes.
    
    Args:
        model: Model to evaluate
        class_images_dict: Dictionary of class names to image paths
        n_way: Number of classes per episode
        k_shot: Number of support examples per class
        q_query: Number of query examples per class
        episodes: Number of episodes to evaluate
        is_training: Whether to use training augmentation (default False for eval)
        device: Device to run evaluation on
        
    Returns:
        mean_loss: Average loss across episodes
        mean_acc: Average accuracy across episodes
        accs: Array of per-episode accuracies
    """
    model.eval()
    losses = []
    accs = []
    
    with torch.no_grad():
        for _ in range(episodes):
            # Sample episode
            support_x, support_y, query_x, query_y = sample_episode(
                class_images_dict,
                n_way,
                k_shot,
                q_query,
                is_training=is_training,
                device=device
            )
            
            # Compute loss and accuracy
            logits = model(support_x, support_y, query_x)
            loss = F.cross_entropy(logits, query_y).item()
            pred = logits.argmax(dim=1)
            acc = (pred == query_y).float().mean().item()
            
            losses.append(loss)
            accs.append(acc)
    
    return np.mean(losses), np.mean(accs), np.array(accs)


def evaluate_all_k_shots(
    model: nn.Module,
    class_images_dict: Dict,
    n_way: int,
    k_shot_list: List[int],
    q_query: int,
    episodes: int = 1000,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model on multiple k-shot settings.
    
    Args:
        model: Model to evaluate
        class_images_dict: Dictionary of class images
        n_way: Number of classes per episode
        k_shot_list: List of k-shot values to evaluate
        q_query: Number of query examples per class
        episodes: Number of episodes per k-shot setting
        device: Device to run on
        
    Returns:
        Dictionary with results for each k-shot setting
    """
    results = {}
    
    for k_shot in k_shot_list:
        print(f"Evaluating {n_way}-way {k_shot}-shot ({episodes} episodes)...")
        
        mean_loss, mean_acc, accs = evaluate_on_episodes(
            model,
            class_images_dict,
            n_way,
            k_shot,
            q_query,
            episodes=episodes,
            is_training=False,
            device=device
        )
        
        results[f"{k_shot}-shot"] = {
            "mean_loss": float(mean_loss),
            "mean_acc": float(mean_acc),
            "std_acc": float(np.std(accs)),
            "all_accs": accs
        }
        
        print(f"  Result: {mean_acc*100:.2f}% ± {np.std(accs)*100:.2f}%")
    
    return results