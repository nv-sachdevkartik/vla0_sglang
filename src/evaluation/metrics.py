"""
Metrics and analysis utilities for model evaluation.
"""

import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute prediction metrics.

    Args:
        predictions: Predicted actions (N, action_dim)
        targets: Target actions (N, action_dim)

    Returns:
        Dictionary of metrics
    """
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # Per-dimension metrics
    per_dim_mse = np.mean((predictions - targets) ** 2, axis=0)
    per_dim_mae = np.mean(np.abs(predictions - targets), axis=0)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
        'per_dim_mse': per_dim_mse.tolist(),
        'per_dim_mae': per_dim_mae.tolist(),
    }


def aggregate_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate multiple evaluation results.

    Args:
        results_list: List of result dictionaries

    Returns:
        Aggregated results dictionary
    """
    if not results_list:
        return {}

    aggregated = {}

    # Aggregate numeric fields
    numeric_fields = ['overall_success_rate', 'mean_episode_length',
                     'mean_inference_time_ms', 'throughput_hz']

    for field in numeric_fields:
        values = [r.get(field, 0) for r in results_list if field in r]
        if values:
            aggregated[f'{field}_mean'] = float(np.mean(values))
            aggregated[f'{field}_std'] = float(np.std(values))
            aggregated[f'{field}_min'] = float(np.min(values))
            aggregated[f'{field}_max'] = float(np.max(values))

    return aggregated
