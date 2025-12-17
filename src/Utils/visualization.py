"""
Visualization utilities for training analysis and model interpretation.

Provides functions to visualize:
- Training curves
- Confusion matrices
- Prototype representations
- Attention maps
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import torch


def plot_learning_curves(
    history_csv: Path,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot training and validation curves.
    
    Args:
        history_csv: Path to history CSV file
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)
    """
    df = pd.read_csv(history_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    axes[0].plot(df['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(df['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(df['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(df['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def visualize_prototypes(
    model,
    class_images_dict: dict,
    n_samples: int = 3,
    save_path: Optional[Path] = None,
    device: str = "cuda"
):
    """
    Visualize prototype representations for sample images.
    
    Args:
        model: Trained model with extract_features method
        class_images_dict: Dictionary of class images
        n_samples: Number of sample classes to visualize
        save_path: Path to save figure (optional)
        device: Device to run model on
    """
    from PIL import Image
    from src.data import load_image
    import random
    
    model.eval()
    
    # Select random classes
    classes = random.sample(
        list(class_images_dict.keys()),
        min(n_samples, len(class_images_dict))
    )
    
    fig, axes = plt.subplots(n_samples, 6, figsize=(15, n_samples * 2.5))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, class_name in enumerate(classes):
            # Load random image from class
            img_path = random.choice(class_images_dict[class_name])
            img_tensor = load_image(img_path, is_training=False).unsqueeze(0).to(device)
            
            # Extract features
            if hasattr(model, 'extract_features'):
                _, prototypes = model.extract_features(img_tensor)
            else:
                print("Model doesn't have extract_features method")
                return
            
            # Show original image
            axes[i, 0].imshow(Image.open(img_path), cmap='gray')
            axes[i, 0].set_title(f"Class: {class_name[:15]}", fontsize=10)
            axes[i, 0].axis('off')
            
            # Visualize prototypes
            prototypes_cpu = prototypes[0].cpu().numpy()
            for j in range(min(5, prototypes.shape[1])):
                # Show first 10 dimensions of each prototype
                axes[i, j + 1].bar(
                    range(min(10, prototypes.shape[2])),
                    prototypes_cpu[j, :10],
                    alpha=0.7
                )
                axes[i, j + 1].set_title(f"Proto {j + 1}", fontsize=9)
                axes[i, j + 1].set_ylim(-3, 3)
                axes[i, j + 1].tick_params(labelsize=7)
    
    plt.suptitle('Prototype Visualizations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prototype visualization to {save_path}")
    
    plt.show()


def plot_results_comparison(
    results_dict: dict,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot comparison of different models/configurations.
    
    Args:
        results_dict: Dictionary with format:
            {
                "model_name": {
                    "k_shot": {"mean_acc": x, "ci_lower": y, "ci_upper": z}
                }
            }
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results_dict.keys())
    x = np.arange(len(models))
    width = 0.35
    
    # Extract 1-shot and 5-shot results
    one_shot_means = []
    one_shot_errors = []
    five_shot_means = []
    five_shot_errors = []
    
    for model in models:
        if "1-shot" in results_dict[model]:
            one_shot = results_dict[model]["1-shot"]
            one_shot_means.append(one_shot["mean_acc"] * 100)
            one_shot_errors.append([
                (one_shot["mean_acc"] - one_shot["ci_lower"]) * 100,
                (one_shot["ci_upper"] - one_shot["mean_acc"]) * 100
            ])
        
        if "5-shot" in results_dict[model]:
            five_shot = results_dict[model]["5-shot"]
            five_shot_means.append(five_shot["mean_acc"] * 100)
            five_shot_errors.append([
                (five_shot["mean_acc"] - five_shot["ci_lower"]) * 100,
                (five_shot["ci_upper"] - five_shot["mean_acc"]) * 100
            ])
    
    # Plot bars
    one_shot_errors = np.array(one_shot_errors).T
    five_shot_errors = np.array(five_shot_errors).T
    
    ax.bar(x - width/2, one_shot_means, width, label='1-shot',
           yerr=one_shot_errors, capsize=5, alpha=0.8)
    ax.bar(x + width/2, five_shot_means, width, label='5-shot',
           yerr=five_shot_errors, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Few-Shot Classification Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved results comparison to {save_path}")
    
    plt.show()


def plot_episode_accuracies(
    accuracies: np.ndarray,
    title: str = "Episode Accuracies",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot distribution and trajectory of episode accuracies.
    
    Args:
        accuracies: Array of per-episode accuracies
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(accuracies * 100, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(accuracies) * 100, color='red', 
                    linestyle='--', linewidth=2, label='Mean')
    axes[0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Accuracy Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Trajectory
    axes[1].plot(accuracies * 100, alpha=0.5, linewidth=0.5)
    axes[1].plot(np.convolve(accuracies * 100, np.ones(100)/100, mode='valid'),
                color='red', linewidth=2, label='Moving Avg (100)')
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Over Episodes', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved episode accuracies plot to {save_path}")