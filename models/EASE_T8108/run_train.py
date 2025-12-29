#!/usr/bin/env python
"""
Training script for EASE model.

Usage:
    # Basic EASE
    python run_train.py --reg_weight 500
    
    # Session-based Weighted EASE
    python run_train.py --weighted --session_threshold 1800 --page_threshold 30 --alpha 0.3
    
    # Grid search
    python run_train.py --grid_search
"""

import itertools
from functools import lru_cache
from typing import Dict, List, Any, Tuple
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from data_loader import DataLoader, WeightedMatrixBuilder
from ease_model import EASE, WeightedEASE
from metrics import evaluate_all, print_metrics
from utils import set_seed, ensure_dir, ExperimentLogger, analyze_time_distribution


def parse_args():
    parser = argparse.ArgumentParser(description='Train EASE model')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='../../data/train/',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Output directory')
    
    # Model parameters
    parser.add_argument('--reg_weight', type=float, default=500.0,
                        help='EASE regularization weight (lambda)')
    
    # Model type selection
    parser.add_argument('--weighted', action='store_true',
                        help='Use session-based weighted EASE')
    
    # Session/Page-based Weighted EASE parameters
    parser.add_argument('--session_threshold', type=float, default=1800.0,
                        help='Session split threshold in seconds (default: 1800 = 30 min)')
    parser.add_argument('--page_threshold', type=float, default=30.0,
                        help='Page split threshold in seconds (default: 30)')
    parser.add_argument('--within_page_weight', type=float, default=1.0,
                        help='Weight for item pairs within same page (default: 1.0)')
    parser.add_argument('--cross_page_tau', type=float, default=60.0,
                        help='Time decay parameter for cross-page pairs in seconds (default: 60)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Weight for session-based matrix (0 = base EASE only)')
    
    # Validation parameters
    parser.add_argument('--valid_random_items', type=int, default=9,
                        help='Number of random items to hold out for validation')
    parser.add_argument('--valid_seq_items', type=int, default=1,
                        help='Number of sequential items to hold out')
    
    # Grid search
    parser.add_argument('--grid_search', action='store_true',
                        help='Run grid search over hyperparameters')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--analyze_time', action='store_true',
                        help='Analyze time distribution in data')
    
    return parser.parse_args()


def train_basic_ease(args, data_loader, train_matrix, valid_matrix, valid_gt, logger):
    """Train basic EASE model."""
    logger.log(f"\n{'='*50}")
    logger.log(f"Training Basic EASE (λ={args.reg_weight})")
    logger.log(f"{'='*50}")
    
    # Fit model
    model = EASE(reg_weight=args.reg_weight)
    model.fit(train_matrix, verbose=True)
    
    # Get recommendations
    logger.log("\nGenerating recommendations...")
    predictions, scores = model.recommend(
        train_matrix,
        top_k=10,
        filter_already_liked=True
    )
    
    # Evaluate
    metrics = evaluate_all(predictions, valid_gt, k_values=[5, 10])
    print_metrics(metrics, "Validation Results")
    
    logger.log_metrics(metrics, {'model': 'EASE', 'reg_weight': args.reg_weight})
    
    return model, metrics


def train_weighted_ease(args, data_loader, train_matrix, valid_matrix, valid_gt, logger):
    """Train session-based weighted EASE model."""
    logger.log(f"\n{'='*50}")
    logger.log(f"Training Session-based Weighted EASE (λ={args.reg_weight})")
    logger.log(f"  Session threshold: {args.session_threshold}s ({args.session_threshold/60:.1f} min)")
    logger.log(f"  Page threshold: {args.page_threshold}s")
    logger.log(f"  Within-page weight: {args.within_page_weight}")
    logger.log(f"  Cross-page tau: {args.cross_page_tau}s")
    logger.log(f"  Alpha: {args.alpha}")
    logger.log(f"{'='*50}")
    
    # Build combined co-occurrence matrix
    builder = WeightedMatrixBuilder(data_loader)
    C_combined = builder.build_combined_cooccurrence_matrix(
        train_matrix=train_matrix,
        alpha=args.alpha,
        session_threshold=args.session_threshold,
        page_threshold=args.page_threshold,
        within_page_weight=args.within_page_weight,
        cross_page_tau=args.cross_page_tau,
        verbose=True
    )
    
    logger.log(f"\nCombined matrix: shape={C_combined.shape}")
    
    # Fit model
    model = WeightedEASE(reg_weight=args.reg_weight)
    model.fit_with_cooccurrence(train_matrix, C_combined, verbose=True)
    
    # Get recommendations
    logger.log("\nGenerating recommendations...")
    predictions, scores = model.recommend(
        train_matrix,
        top_k=10,
        filter_already_liked=True
    )
    
    # Evaluate
    metrics = evaluate_all(predictions, valid_gt, k_values=[5, 10])
    print_metrics(metrics, "Validation Results")
    
    logger.log_metrics(metrics, {
        'model': 'SessionWeightedEASE',
        'reg_weight': args.reg_weight,
        'session_threshold': args.session_threshold,
        'page_threshold': args.page_threshold,
        'within_page_weight': args.within_page_weight,
        'cross_page_tau': args.cross_page_tau,
        'alpha': args.alpha
    })
    
    return model, metrics


def run_grid_search(args, data_loader, train_matrix, valid_matrix, valid_gt, logger):
    """Run grid search over hyperparameters with flexible fixed/search parameter switching."""
    logger.log("\n" + "="*50)
    logger.log("Starting Grid Search")
    logger.log("="*50)
    
    search_space = {
        'reg_weight': [4800.0],
        'session_threshold': [1500.0, 1800.0, 2100.0],
        'page_threshold': [4],
        'alpha': [49, 50, 51],
        'cross_page_tau': [550.0, 560.0, 570.0, 580.0],
        'within_page_weight': [0.2, 0.25, 0.3],
    }
    
    # 파라미터 그룹 정의: 어떤 파라미터가 바뀌면 어떤 계산을 다시 해야 하는지
    MATRIX_BUILD_PARAMS = {'session_threshold', 'page_threshold', 'cross_page_tau', 'within_page_weight'}
    
    # =================================================================
    # 캐싱을 위한 헬퍼 함수들
    # =================================================================
    builder = WeightedMatrixBuilder(data_loader)
    
    # C_base는 train_matrix에만 의존하므로 한 번만 계산
    C_base = (train_matrix.T @ train_matrix).toarray().astype(np.float32)
    base_mean = C_base[C_base > 0].mean()
    
    # Session matrix 캐시 (dictionary 기반, lru_cache는 unhashable 인자 문제)
    session_matrix_cache: Dict[Tuple, np.ndarray] = {}
    
    def get_session_matrix(session_threshold: float, page_threshold: int, 
                           cross_page_tau: float, within_page_weight: float) -> np.ndarray:
        """캐싱된 session cooccurrence matrix 반환."""
        cache_key = (session_threshold, page_threshold, cross_page_tau, within_page_weight)
        
        if cache_key not in session_matrix_cache:
            logger.log(f"\n[Building matrix] session_th={session_threshold}s, page_th={page_threshold}s, "
                      f"tau={cross_page_tau}s, within_weight={within_page_weight}")
            
            C_session = builder.build_session_cooccurrence_matrix(
                session_threshold=session_threshold,
                page_threshold=page_threshold,
                within_page_weight=within_page_weight,
                cross_page_tau=cross_page_tau,
                train_matrix=train_matrix,
                verbose=True
            ).toarray().astype(np.float32)
            
            # Normalize
            session_max = C_session.max()
            if session_max > 0:
                C_session = C_session / session_max
            
            session_matrix_cache[cache_key] = C_session
        
        return session_matrix_cache[cache_key]
    
    # =================================================================
    # Grid Search 실행
    # =================================================================
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_combinations = list(itertools.product(*param_values))
    total_combinations = len(all_combinations)
    
    # 로깅: 고정 vs 탐색 파라미터 구분
    fixed_params = {k: v[0] for k, v in search_space.items() if len(v) == 1}
    search_params = {k: v for k, v in search_space.items() if len(v) > 1}
    
    logger.log(f"\nFixed parameters: {fixed_params}")
    logger.log(f"Search parameters: {list(search_params.keys())}")
    logger.log(f"Grid sizes: {' × '.join(f'{len(v)}' for v in search_params.values())}")
    logger.log(f"Total combinations: {total_combinations}")
    
    best_recall = 0
    best_params = {}
    best_model = None
    
    for idx, param_tuple in enumerate(all_combinations, 1):
        config = dict(zip(param_names, param_tuple))
        
        C_session_norm = get_session_matrix(
            session_threshold=config['session_threshold'],
            page_threshold=config['page_threshold'],
            cross_page_tau=config['cross_page_tau'],
            within_page_weight=config['within_page_weight']
        )
        
        C_combined = C_base + config['alpha'] * base_mean * C_session_norm
        
        model = WeightedEASE(reg_weight=config['reg_weight'])
        model.fit_with_cooccurrence(train_matrix, C_combined, verbose=False)
        
        predictions, _ = model.recommend(train_matrix, top_k=10, filter_already_liked=True)
        metrics = evaluate_all(predictions, valid_gt, k_values=[10])
        
        recall_10 = metrics['Recall@10']
        
        log_params = {'model': 'SessionWeightedEASE', **config}
        logger.log_metrics(metrics, log_params)
        
        is_best = recall_10 > best_recall
        if is_best:
            best_recall = recall_10
            best_params = log_params.copy()
            best_model = model
        
        search_vals = ' | '.join(f"{k}={config[k]}" for k in search_params.keys())
        status = "★ BEST" if is_best else ""
        logger.log(f"  [{idx}/{total_combinations}] {search_vals} → Recall@10={recall_10:.4f} {status}")
    
    logger.log("\n" + "="*50)
    logger.log("Grid Search Complete!")
    logger.log(f"Best Recall@10: {best_recall:.4f}")
    logger.log(f"Best Parameters: {best_params}")
    logger.log(f"Session matrix cache hits: {len(session_matrix_cache)} unique matrices built")
    logger.log("="*50)
    
    return best_model, best_params


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    
    # Initialize logger
    if args.grid_search:
        experiment_name = "grid_search"
    elif args.weighted:
        experiment_name = "session_weighted_ease"
    else:
        experiment_name = "ease"
    logger = ExperimentLogger(args.output_dir, experiment_name)
    
    logger.log("EASE Model Training")
    logger.log_params(vars(args))
    
    # Load data
    logger.log("\nLoading data...")
    data_file = os.path.join(args.data_dir, 'train_ratings.csv')
    data_loader = DataLoader(data_file, seed=args.seed)
    
    # Analyze time distribution if requested
    if args.analyze_time:
        time_stats = analyze_time_distribution(data_loader)
    
    # Create train/valid split
    train_matrix, valid_matrix, valid_gt = data_loader.create_train_valid_split(
        valid_random_items=args.valid_random_items,
        valid_seq_items=args.valid_seq_items
    )
    
    logger.log(f"\nTrain matrix: {train_matrix.shape}, nnz={train_matrix.nnz}")
    
    # Train model
    if args.grid_search:
        model, best_params = run_grid_search(
            args, data_loader, train_matrix, valid_matrix, valid_gt, logger
        )
    elif args.weighted:
        model, metrics = train_weighted_ease(
            args, data_loader, train_matrix, valid_matrix, valid_gt, logger
        )
    else:
        model, metrics = train_basic_ease(
            args, data_loader, train_matrix, valid_matrix, valid_gt, logger
        )
    
    # Save model weights
    if model is not None:
        model_path = os.path.join(args.output_dir, 'ease_model.npy')
        np.save(model_path, model.B)
        logger.log(f"\nModel saved to: {model_path}")
    
    logger.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()