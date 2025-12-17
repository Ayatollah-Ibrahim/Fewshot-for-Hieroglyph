"""Utility functions for saving, loading, and visualization."""

from src.utils.metrics import bootstrap_ci, compute_confidence_interval
from src.utils.visualization import (
    plot_learning_curves,
    plot_confusion_matrix,
    visualize_prototypes
)

import json
import pandas as pd
from pathlib import Path


def save_json(obj: dict, path: Path):
    """
    Save dictionary to JSON file.
    
    Args:
        obj: Dictionary to save
        path: Output path
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> dict:
    """
    Load JSON file to dictionary.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path):
    """
    Save DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        path: Output path
    """
    df.to_csv(path, index=False)


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load CSV to DataFrame.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(path)


__all__ = [
    "bootstrap_ci",
    "compute_confidence_interval",
    "plot_learning_curves",
    "plot_confusion_matrix",
    "visualize_prototypes",
    "save_json",
    "load_json",
    "save_csv",
    "load_csv",
]