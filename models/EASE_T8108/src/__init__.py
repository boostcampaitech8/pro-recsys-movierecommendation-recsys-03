"""
EASE Model Package for Movie Recommendation.

Session/Page-based Weighted EASE implementation.

Hierarchy:
    User Sequence → Session (30min gap) → Page (30sec gap) → Items

Weight rules:
    - Same page: within_page_weight (default 1.0)
    - Same session, different page: exp(-Δt / cross_page_tau)
    - Different session: 0 (not computed)
"""

from .data_loader import DataLoader, WeightedMatrixBuilder
from .ease_model import EASE, WeightedEASE
from .metrics import recall_at_k, ndcg_at_k, evaluate_all, print_metrics
from .utils import set_seed, load_config, create_submission, ExperimentLogger

__all__ = [
    'DataLoader',
    'WeightedMatrixBuilder', 
    'EASE',
    'WeightedEASE',
    'recall_at_k',
    'ndcg_at_k',
    'evaluate_all',
    'print_metrics',
    'set_seed',
    'load_config',
    'create_submission',
    'ExperimentLogger',
]