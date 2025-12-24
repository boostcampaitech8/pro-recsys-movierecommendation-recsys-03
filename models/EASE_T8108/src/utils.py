"""
Utility functions for EASE model.
"""

import os
import yaml
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_dir(directory: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_submission(
    predictions: np.ndarray,
    idx2user: Dict[int, int],
    idx2item: Dict[int, int],
    output_path: str,
    top_k: int = 10
):
    """
    Create submission CSV file.
    
    Args:
        predictions: Array of predicted item indices, shape (n_users, top_k)
        idx2user: Mapping from internal user index to original user ID
        idx2item: Mapping from internal item index to original item ID
        output_path: Path to save submission file
        top_k: Number of items per user
    """
    print(f"Creating submission file: {output_path}")
    
    rows = []
    n_users = predictions.shape[0]
    
    for user_idx in range(n_users):
        user_id = idx2user[user_idx]
        for item_idx in predictions[user_idx, :top_k]:
            item_id = idx2item[item_idx]
            rows.append({'user': user_id, 'item': item_id})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"  Saved {len(df)} rows ({n_users} users x {top_k} items)")
    return df


def analyze_time_distribution(data_loader, sample_users: int = 1000):
    """
    Analyze time difference distribution in user sequences.
    
    Args:
        data_loader: DataLoader instance
        sample_users: Number of users to sample for analysis
        
    Returns:
        Dictionary with analysis results
    """
    print("Analyzing time difference distribution...")
    
    all_deltas = []
    user_indices = list(data_loader.user_sequences.keys())
    
    if len(user_indices) > sample_users:
        user_indices = random.sample(user_indices, sample_users)
    
    for user_idx in user_indices:
        seq = data_loader.user_sequences[user_idx]
        if len(seq) < 2:
            continue
        
        times = [t for _, t in seq]
        deltas = [times[i+1] - times[i] for i in range(len(times)-1)]
        all_deltas.extend(deltas)
    
    all_deltas = np.array(all_deltas)
    
    results = {
        'n_deltas': len(all_deltas),
        'mean': np.mean(all_deltas),
        'median': np.median(all_deltas),
        'std': np.std(all_deltas),
        'min': np.min(all_deltas),
        'max': np.max(all_deltas),
        'percentiles': {
            '1%': np.percentile(all_deltas, 1),
            '5%': np.percentile(all_deltas, 5),
            '10%': np.percentile(all_deltas, 10),
            '25%': np.percentile(all_deltas, 25),
            '50%': np.percentile(all_deltas, 50),
            '75%': np.percentile(all_deltas, 75),
            '90%': np.percentile(all_deltas, 90),
            '95%': np.percentile(all_deltas, 95),
            '99%': np.percentile(all_deltas, 99),
        },
        'short_interval_counts': {
            '<=3s': np.sum(all_deltas <= 3),
            '<=5s': np.sum(all_deltas <= 5),
            '<=10s': np.sum(all_deltas <= 10),
            '<=30s': np.sum(all_deltas <= 30),
            '<=60s': np.sum(all_deltas <= 60),
        }
    }
    
    print(f"\nTime Difference Statistics:")
    print(f"  Total intervals: {results['n_deltas']}")
    print(f"  Mean: {results['mean']:.1f}s, Median: {results['median']:.1f}s")
    print(f"  Min: {results['min']:.1f}s, Max: {results['max']:.1f}s")
    print(f"\n  Short interval counts:")
    for label, count in results['short_interval_counts'].items():
        pct = 100 * count / results['n_deltas']
        print(f"    {label}: {count} ({pct:.1f}%)")
    
    return results


class ExperimentLogger:
    """Simple logger for tracking experiments."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        ensure_dir(log_dir)
        self.log_path = os.path.join(log_dir, f"{experiment_name}_{get_timestamp()}.log")
        self.results_path = os.path.join(log_dir, f"{experiment_name}_{get_timestamp()}_results.csv")
        self.results = []
        
        with open(self.log_path, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
    
    def log(self, message: str):
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_path, 'a') as f:
            f.write(log_line + "\n")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        self.log("Parameters:")
        for key, value in params.items():
            self.log(f"  {key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, float], params: Optional[Dict[str, Any]] = None):
        """Log metrics and optionally save to results CSV."""
        self.log("Metrics:")
        for key, value in metrics.items():
            self.log(f"  {key}: {value:.4f}")
        
        # Save to results list
        result_row = metrics.copy()
        if params:
            result_row.update(params)
        self.results.append(result_row)
        
        # Save to CSV
        pd.DataFrame(self.results).to_csv(self.results_path, index=False)
    
    def finish(self):
        """Finish logging."""
        self.log(f"\nExperiment finished: {datetime.now()}")


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    print("Seed set successfully")
    
    timestamp = get_timestamp()
    print(f"Timestamp: {timestamp}")
