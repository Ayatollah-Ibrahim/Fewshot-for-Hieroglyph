"""
Statistical metrics and confidence interval computation.

Provides bootstrap-based confidence intervals for evaluating
few-shot learning performance.
"""

import numpy as np
from typing import Tuple


def bootstrap_ci(
    acc_array: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for accuracy.
    
    Args:
        acc_array: Array of per-episode accuracies
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        
    Returns:
        mean: Mean accuracy
        lower: Lower confidence bound
        upper: Upper confidence bound
    """
    n = len(acc_array)
    means = []
    
    # Bootstrap sampling
    for _ in range(n_bootstrap):
        sample = np.random.choice(acc_array, size=n, replace=True)
        means.append(sample.mean())
    
    # Compute percentiles
    alpha = 1 - ci
    lower = np.percentile(means, (alpha / 2) * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)
    
    return np.mean(means), lower, upper


def compute_confidence_interval(
    acc_array: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval using normal approximation.
    
    Args:
        acc_array: Array of accuracies
        confidence: Confidence level
        
    Returns:
        mean: Mean accuracy
        lower: Lower confidence bound
        upper: Upper confidence bound
    """
    import scipy.stats as stats
    
    mean = np.mean(acc_array)
    std_err = stats.sem(acc_array)
    
    # Compute margin of error
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(acc_array) - 1)
    
    lower = mean - margin
    upper = mean + margin
    
    return mean, lower, upper


def compute_accuracy_stats(acc_array: np.ndarray) -> dict:
    """
    Compute comprehensive accuracy statistics.
    
    Args:
        acc_array: Array of accuracies
        
    Returns:
        Dictionary with statistical measures
    """
    return {
        "mean": float(np.mean(acc_array)),
        "std": float(np.std(acc_array)),
        "min": float(np.min(acc_array)),
        "max": float(np.max(acc_array)),
        "median": float(np.median(acc_array)),
        "q25": float(np.percentile(acc_array, 25)),
        "q75": float(np.percentile(acc_array, 75)),
    }

