"""Training and evaluation utilities for few-shot learning."""

from src.training.trainer import Trainer, mixup_data, mixup_criterion
from src.training.evaluation import evaluate_on_episodes, episode_loss_and_acc

__all__ = [
    "Trainer",
    "mixup_data",
    "mixup_criterion",
    "evaluate_on_episodes",
    "episode_loss_and_acc",
]