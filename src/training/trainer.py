"""
Training utilities and trainer class for meta-learning.

Implements episodic training loop with support for:
- Mixup augmentation
- Label smoothing
- Early stopping
- Learning rate scheduling
- Gradient clipping
"""

import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.data import sample_episode
from src.training.evaluation import evaluate_on_episodes
from src.utils import save_json, save_csv


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation to input data.
    
    Mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
    
    Args:
        x: Input images [B, C, H, W]
        y: Labels [B]
        alpha: Beta distribution parameter for mixing coefficient
        
    Returns:
        mixed_x: Mixed images
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute loss for mixup-augmented data.
    
    Args:
        pred: Model predictions [B, C]
        y_a: Original labels [B]
        y_b: Mixed labels [B]
        lam: Mixing coefficient
        
    Returns:
        Mixed cross-entropy loss
    """
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)


class Trainer:
    """
    Trainer for meta-learning models.
    
    Handles episodic training with various regularization techniques
    and automatic checkpoint saving.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict,
        save_dir: Path,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            config: Configuration dictionary
            save_dir: Directory to save checkpoints and logs
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.save_dir = Path(save_dir)
        self.device = device
        
        self.best_val_acc = -1.0
        self.epochs_no_improve = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(
        self,
        train_class_images: Dict,
        use_mixup: bool = True,
        mixup_prob: float = 0.3,
        label_smoothing: float = 0.1
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_class_images: Dictionary of class names to image paths
            use_mixup: Whether to use mixup augmentation
            mixup_prob: Probability of applying mixup
            label_smoothing: Amount of label smoothing
            
        Returns:
            mean_loss: Average loss over epoch
            mean_acc: Average accuracy over epoch
        """
        self.model.train()
        epoch_losses = []
        epoch_accs = []
        
        n_way = self.config["N_WAY"]
        k_shot = self.config["K_SHOT_LIST"][-1]  # Use largest k-shot
        q_query = self.config["Q_QUERY"]
        episodes_per_epoch = self.config["EPISODES_PER_EPOCH"]
        meta_batch = self.config["META_BATCH_SIZE"]
        
        for ep in range(episodes_per_epoch):
            self.optimizer.zero_grad()
            batch_loss = 0.0
            batch_acc = 0.0
            
            for _ in range(meta_batch):
                # Sample episode with training augmentation
                support_x, support_y, query_x, query_y = sample_episode(
                    train_class_images,
                    n_way,
                    k_shot,
                    q_query,
                    is_training=True,
                    device=self.device
                )
                
                # Apply mixup with probability
                if use_mixup and random.random() < mixup_prob:
                    query_x, query_y_a, query_y_b, lam = mixup_data(
                        query_x, query_y, alpha=0.2
                    )
                    logits = self.model(support_x, support_y, query_x)
                    loss = mixup_criterion(logits, query_y_a, query_y_b, lam)
                    
                    # Compute accuracy
                    pred = logits.argmax(dim=1)
                    acc = (lam * (pred == query_y_a).float() + 
                          (1-lam) * (pred == query_y_b).float()).mean().item()
                else:
                    logits = self.model(support_x, support_y, query_x)
                    
                    # Label smoothing
                    if label_smoothing > 0:
                        n_classes = logits.size(1)
                        with torch.no_grad():
                            true_dist = torch.zeros_like(logits)
                            true_dist.fill_(label_smoothing / (n_classes - 1))
                            true_dist.scatter_(1, query_y.unsqueeze(1), 1.0 - label_smoothing)
                        loss = (-true_dist * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(logits, query_y)
                    
                    # Compute accuracy
                    pred = logits.argmax(dim=1)
                    acc = (pred == query_y).float().mean().item()
                
                loss.backward()
                batch_loss += loss.item()
                batch_acc += acc
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            epoch_losses.append(batch_loss / meta_batch)
            epoch_accs.append(batch_acc / meta_batch)
            
            # Print progress
            if (ep + 1) % self.config["PRINT_EVERY"] == 0:
                avg_loss = batch_loss / meta_batch
                avg_acc = batch_acc / meta_batch
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Ep {ep+1}/{episodes_per_epoch} - "
                      f"loss {avg_loss:.4f} - acc {avg_acc:.4f} - lr {current_lr:.6f}")
        
        return np.mean(epoch_losses), np.mean(epoch_accs)
    
    def validate(
        self,
        val_class_images: Dict,
        episodes: int = 500
    ) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_class_images: Dictionary of validation class images
            episodes: Number of episodes to evaluate
            
        Returns:
            mean_loss: Average validation loss
            mean_acc: Average validation accuracy
        """
        n_way = self.config["N_WAY"]
        k_shot = self.config["K_SHOT_LIST"][-1]
        q_query = self.config["Q_QUERY"]
        
        val_loss, val_acc, _ = evaluate_on_episodes(
            self.model,
            val_class_images,
            n_way,
            k_shot,
            q_query,
            episodes=episodes,
            is_training=False,
            device=self.device
        )
        
        return val_loss, val_acc
    
    def train(
        self,
        train_class_images: Dict,
        val_class_images: Dict
    ):
        """
        Full training loop with early stopping.
        
        Args:
            train_class_images: Training class images
            val_class_images: Validation class images
        """
        max_epochs = self.config["MAX_EPOCHS"]
        patience = self.config["PATIENCE"]
        val_episodes = self.config["VAL_EPISODES"]
        
        print(f"\n{'='*60}")
        print(f"Training {self.config['MODEL_NAME']}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, max_epochs + 1):
            start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_class_images)
            
            # Validate
            val_loss, val_acc = self.validate(val_class_images, val_episodes)
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            
            elapsed = time.time() - start
            gap = train_acc - val_acc
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} Summary ({elapsed/60:.2f} min):")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
            print(f"  Gap: {gap:.4f}, LR: {current_lr:.6f}")
            print(f"{'='*60}\n")
            
            # Early stopping and checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"  ✅ New best validation: {self.best_val_acc:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"  ⏳ No improvement for {self.epochs_no_improve} epoch(s)")
                
                if self.epochs_no_improve >= patience:
                    print("  🛑 Early stopping triggered.")
                    break
            
            # Save periodic checkpoint
            if epoch % self.config.get("SAVE_EVERY_EPOCHS", 5) == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)
        
        # Save training history
        self.save_history()
        print(f"\n✅ Training finished. Best val acc: {self.best_val_acc:.4f}")
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
        }
        
        if is_best:
            path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, path)
            save_json({"best_val": float(self.best_val_acc), "epoch": epoch},
                     self.save_dir / "best_meta.json")
        else:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to CSV."""
        df = pd.DataFrame(self.history)
        save_csv(df, self.save_dir / "history.csv")
