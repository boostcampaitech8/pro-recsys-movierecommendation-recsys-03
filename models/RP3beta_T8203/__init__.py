"""
RP3beta recommendation model package.

This package contains:
- config: experiment configuration
- data: data loading and preprocessing
- model: RP3beta model implementation
- trainer: training & offline validation
- evaluate: evaluation metrics
- make_submission: submission file generation
"""

from .config import *
from .data import InteractionData
from .model import RP3beta
from .trainer import RP3Trainer
