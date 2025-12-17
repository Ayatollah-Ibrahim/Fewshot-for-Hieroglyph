"""
Egyptian Hieroglyphic Few-Shot Learning Package

A comprehensive framework for few-shot classification of hieroglyphic glyphs
using meta-learning approaches including HPGN and ProtoNet.
"""

__version__ = "1.0.0"
__author__ = "Aya"

from src.models import HPGN_Small, ProtoNet
from src.data import (
    ClassFolderDataset,
    get_train_transform,
    get_val_transform,
    sample_episode
)
from src.training import Trainer, evaluate_on_episodes
from src.utils import bootstrap_ci, save_json, save_csv

__all__ = [
    "HPGN_Small",
    "ProtoNet",
    "ClassFolderDataset",
    "get_train_transform",
    "get_val_transform",
    "sample_episode",
    "Trainer",
    "evaluate_on_episodes",
    "bootstrap_ci",
    "save_json",
    "save_csv",
]
