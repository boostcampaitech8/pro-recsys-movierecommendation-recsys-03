"""
Evaluation metrics for recommendation systems.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


def recall_at_k(
    predictions: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Compute Recall@K.
    
    Recall@K = |{relevant items in top-K}| / |{relevant items}|
    
    Args:
        predictions: Array of predicted item indices, shape (n_users, top_k)
        ground_truth: Dict mapping user_idx to list of relevant item indices
        k: Number of recommendations to consider
        
    Returns:
        Average Recall@K across all users
    """
    total_recall = 0.0
    n_users_with_gt = 0
    
    for user_idx, true_items in ground_truth.items():
        if len(true_items) == 0:
            continue
        
        pred_items = set(predictions[user_idx, :k])
        true_items_set = set(true_items)
        
        n_hits = len(pred_items & true_items_set)
        recall = n_hits / len(true_items_set)
        
        total_recall += recall
        n_users_with_gt += 1
    
    if n_users_with_gt == 0:
        return 0.0
    
    return total_recall / n_users_with_gt


def ndcg_at_k(
    predictions: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).
    
    Args:
        predictions: Array of predicted item indices
        ground_truth: Dict mapping user_idx to list of relevant item indices
        k: Number of recommendations
        
    Returns:
        Average NDCG@K across all users
    """
    total_ndcg = 0.0
    n_users_with_gt = 0
    
    for user_idx, true_items in ground_truth.items():
        if len(true_items) == 0:
            continue
        
        true_items_set = set(true_items)
        pred_items = predictions[user_idx, :k]
        
        # DCG
        dcg = 0.0
        for i, item in enumerate(pred_items):
            if item in true_items_set:
                dcg += 1.0 / np.log2(i + 2)  # position is 1-indexed
        
        # IDCG (ideal DCG)
        n_relevant = min(len(true_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
        
        if idcg > 0:
            total_ndcg += dcg / idcg
        
        n_users_with_gt += 1
    
    if n_users_with_gt == 0:
        return 0.0
    
    return total_ndcg / n_users_with_gt


def hit_rate_at_k(
    predictions: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Compute Hit Rate@K (proportion of users with at least one hit).
    
    Args:
        predictions: Array of predicted item indices
        ground_truth: Dict mapping user_idx to list of relevant item indices
        k: Number of recommendations
        
    Returns:
        Hit rate
    """
    n_hits = 0
    n_users_with_gt = 0
    
    for user_idx, true_items in ground_truth.items():
        if len(true_items) == 0:
            continue
        
        pred_items = set(predictions[user_idx, :k])
        true_items_set = set(true_items)
        
        if len(pred_items & true_items_set) > 0:
            n_hits += 1
        
        n_users_with_gt += 1
    
    if n_users_with_gt == 0:
        return 0.0
    
    return n_hits / n_users_with_gt


def precision_at_k(
    predictions: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Compute Precision@K.
    
    Args:
        predictions: Array of predicted item indices
        ground_truth: Dict mapping user_idx to list of relevant item indices
        k: Number of recommendations
        
    Returns:
        Average Precision@K
    """
    total_precision = 0.0
    n_users_with_gt = 0
    
    for user_idx, true_items in ground_truth.items():
        if len(true_items) == 0:
            continue
        
        pred_items = set(predictions[user_idx, :k])
        true_items_set = set(true_items)
        
        n_hits = len(pred_items & true_items_set)
        precision = n_hits / k
        
        total_precision += precision
        n_users_with_gt += 1
    
    if n_users_with_gt == 0:
        return 0.0
    
    return total_precision / n_users_with_gt


def evaluate_all(
    predictions: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute all metrics for given K values.
    
    Args:
        predictions: Array of predicted item indices
        ground_truth: Dict mapping user_idx to list of relevant item indices
        k_values: List of K values to evaluate
        
    Returns:
        Dictionary of metric_name -> value
    """
    results = {}
    
    for k in k_values:
        results[f'Recall@{k}'] = recall_at_k(predictions, ground_truth, k)
        results[f'NDCG@{k}'] = ndcg_at_k(predictions, ground_truth, k)
        results[f'Precision@{k}'] = precision_at_k(predictions, ground_truth, k)
        results[f'HitRate@{k}'] = hit_rate_at_k(predictions, ground_truth, k)
    
    return results


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    # Group by metric type
    metric_types = defaultdict(dict)
    for name, value in metrics.items():
        metric_name, k = name.rsplit('@', 1)
        metric_types[metric_name][int(k)] = value
    
    for metric_name, k_values in metric_types.items():
        print(f"\n{metric_name}:")
        for k in sorted(k_values.keys()):
            print(f"  @{k}: {k_values[k]:.4f}")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Dummy predictions and ground truth
    n_users = 100
    n_items = 50
    top_k = 10
    
    predictions = np.random.randint(0, n_items, size=(n_users, top_k))
    ground_truth = {i: list(np.random.choice(n_items, size=3, replace=False)) 
                    for i in range(n_users)}
    
    metrics = evaluate_all(predictions, ground_truth, k_values=[5, 10])
    print_metrics(metrics, "Test Metrics")
