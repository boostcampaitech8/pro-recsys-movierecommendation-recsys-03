#!/usr/bin/env python
"""
Training script for EASE model.

Usage:
    # Basic EASE
    python run_train.py --reg_weight 500
    
    # Weighted EASE with time decay
    python run_train.py --weighted --tau 30 --reg_weight 500
    
    # Grid search
    python run_train.py --grid_search
"""

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
                        help='Use weighted EASE (hybrid window, time-only)')
    parser.add_argument('--combined', action='store_true',
                        help='Use combined EASE (base + time weighted)')
    
    # Combined EASE parameters
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Weight for time-based matrix in combined mode (0=base only)')
    parser.add_argument('--scale', type=float, default=None,
                        help='Scale factor for time matrix (None=auto, uses mean of base matrix)')
    
    # Time-weighted parameters (for both weighted and combined)
    parser.add_argument('--max_window_size', type=int, default=50,
                        help='Maximum window size by count (default: 50)')
    parser.add_argument('--max_time_diff', type=float, default=60.0,
                        help='Maximum time difference in seconds (default: 60)')
    parser.add_argument('--weight_mode', type=str, default='exponential',
                        choices=['exponential', 'binary'],
                        help='Weight mode: exponential or binary')
    parser.add_argument('--tau', type=float, default=10.0,
                        help='Time decay parameter for exponential mode (seconds)')
    parser.add_argument('--time_threshold', type=float, default=5.0,
                        help='Threshold for binary mode (seconds)')
    parser.add_argument('--session_threshold', type=float, default=None,
                        help='Session separation threshold (seconds), None to disable')
    parser.add_argument('--stride', type=int, default=1,
                        help='Window sliding step (default: 1)')
    
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
    """Train weighted EASE model with hybrid window co-occurrence."""
    logger.log(f"\n{'='*50}")
    logger.log(f"Training Weighted EASE (λ={args.reg_weight})")
    logger.log(f"  Max window size: {args.max_window_size}")
    logger.log(f"  Max time diff: {args.max_time_diff}s")
    logger.log(f"  Weight mode: {args.weight_mode}")
    if args.weight_mode == 'exponential':
        logger.log(f"  Tau: {args.tau}s")
    elif args.weight_mode == 'binary':
        logger.log(f"  Time threshold: {args.time_threshold}s")
    if args.session_threshold:
        logger.log(f"  Session threshold: {args.session_threshold}s")
    logger.log(f"  Stride: {args.stride}")
    logger.log(f"{'='*50}")
    
    # Build hybrid co-occurrence matrix
    builder = WeightedMatrixBuilder(data_loader)
    C = builder.build_hybrid_cooccurrence_matrix(
        max_window_size=args.max_window_size,
        max_time_diff=args.max_time_diff,
        weight_mode=args.weight_mode,
        tau=args.tau,
        time_threshold=args.time_threshold,
        session_threshold=args.session_threshold,
        stride=args.stride,
        train_matrix=train_matrix,
        verbose=True
    )
    
    logger.log(f"Co-occurrence matrix: shape={C.shape}, nnz={C.nnz}")
    
    # Fit model
    model = WeightedEASE(reg_weight=args.reg_weight)
    model.fit_with_cooccurrence(train_matrix, C, verbose=True)
    
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
        'model': 'HybridEASE',
        'reg_weight': args.reg_weight,
        'max_window_size': args.max_window_size,
        'max_time_diff': args.max_time_diff,
        'weight_mode': args.weight_mode,
        'tau': args.tau if args.weight_mode == 'exponential' else None,
        'time_threshold': args.time_threshold if args.weight_mode == 'binary' else None,
        'session_threshold': args.session_threshold,
        'stride': args.stride
    })
    
    return model, metrics


def train_combined_ease(args, data_loader, train_matrix, valid_matrix, valid_gt, logger):
    """Train combined EASE model (base + time-weighted, Additive approach)."""
    logger.log(f"\n{'='*50}")
    logger.log(f"Training Combined EASE (λ={args.reg_weight})")
    logger.log(f"  Alpha (time weight): {args.alpha}")
    logger.log(f"  Scale: {args.scale if args.scale else 'auto'}")
    logger.log(f"  Weight mode: {args.weight_mode}")
    if args.weight_mode == 'exponential':
        logger.log(f"  Tau: {args.tau}s")
    elif args.weight_mode == 'binary':
        logger.log(f"  Time threshold: {args.time_threshold}s")
    logger.log(f"  Max time diff: {args.max_time_diff}s")
    logger.log(f"{'='*50}")
    
    # Build combined co-occurrence matrix
    builder = WeightedMatrixBuilder(data_loader)
    C_combined = builder.build_combined_cooccurrence_matrix(
        train_matrix=train_matrix,
        alpha=args.alpha,
        scale=args.scale,
        max_time_diff=args.max_time_diff,
        weight_mode=args.weight_mode,
        tau=args.tau,
        time_threshold=args.time_threshold,
        max_window_size=args.max_window_size,
        verbose=True
    )
    
    logger.log(f"Combined matrix: shape={C_combined.shape}")
    
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
        'model': 'CombinedEASE',
        'reg_weight': args.reg_weight,
        'alpha': args.alpha,
        'scale': args.scale,
        'weight_mode': args.weight_mode,
        'tau': args.tau if args.weight_mode == 'exponential' else None,
        'time_threshold': args.time_threshold if args.weight_mode == 'binary' else None,
        'max_time_diff': args.max_time_diff
    })
    
    return model, metrics


def run_grid_search(args, data_loader, train_matrix, valid_matrix, valid_gt, logger):
    """Run grid search over hyperparameters."""
    logger.log("\n" + "="*50)
    logger.log("Starting Grid Search")
    logger.log("="*50)
    
    # Hyperparameter grid
    reg_weights = [250, 500, 1000, 1500]
    
    best_recall = 0
    best_params = {}
    best_model = None
    
    # Basic EASE grid search
    logger.log("\n--- Basic EASE ---")
    for reg_weight in reg_weights:
        logger.log(f"\nTrying λ={reg_weight}...")
        
        model = EASE(reg_weight=reg_weight)
        model.fit(train_matrix, verbose=False)
        
        predictions, _ = model.recommend(train_matrix, top_k=10, filter_already_liked=True)
        metrics = evaluate_all(predictions, valid_gt, k_values=[10])
        
        recall_10 = metrics['Recall@10']
        logger.log(f"  Recall@10: {recall_10:.4f}")
        
        logger.log_metrics(metrics, {'model': 'EASE', 'reg_weight': reg_weight})
        
        if recall_10 > best_recall:
            best_recall = recall_10
            best_params = {'model': 'EASE', 'reg_weight': reg_weight}
            best_model = model
    
    # Hybrid EASE grid search
    logger.log("\n--- Hybrid EASE ---")
    
    # Grid parameters
    max_time_diffs = [30, 60, 120, 300]
    weight_modes = ['exponential', 'binary']
    taus = [10, 30, 60]  # for exponential
    time_thresholds = [5, 10, 30]  # for binary
    
    builder = WeightedMatrixBuilder(data_loader)
    
    for max_time_diff in max_time_diffs:
        for weight_mode in weight_modes:
            if weight_mode == 'exponential':
                weight_params = taus
                param_name = 'tau'
            else:  # binary
                weight_params = time_thresholds
                param_name = 'time_threshold'
            
            for weight_param in weight_params:
                logger.log(f"\nBuilding co-occurrence matrix (max_time_diff={max_time_diff}, mode={weight_mode}, {param_name}={weight_param})...")
                
                C = builder.build_hybrid_cooccurrence_matrix(
                    max_window_size=50,  # Large enough to be time-dominant
                    max_time_diff=max_time_diff,
                    weight_mode=weight_mode,
                    tau=weight_param if weight_mode == 'exponential' else 30.0,
                    time_threshold=weight_param if weight_mode == 'binary' else 10.0,
                    session_threshold=None,
                    stride=1,
                    train_matrix=train_matrix,
                    verbose=False
                )
                
                for reg_weight in reg_weights:
                    logger.log(f"  Trying λ={reg_weight}...")
                    
                    model = WeightedEASE(reg_weight=reg_weight)
                    model.fit_with_cooccurrence(train_matrix, C, verbose=False)
                    
                    predictions, _ = model.recommend(train_matrix, top_k=10, filter_already_liked=True)
                    metrics = evaluate_all(predictions, valid_gt, k_values=[10])
                    
                    recall_10 = metrics['Recall@10']
                    logger.log(f"    Recall@10: {recall_10:.4f}")
                    
                    log_params = {
                        'model': 'HybridEASE',
                        'reg_weight': reg_weight,
                        'max_time_diff': max_time_diff,
                        'weight_mode': weight_mode,
                    }
                    if weight_mode == 'exponential':
                        log_params['tau'] = weight_param
                    else:
                        log_params['time_threshold'] = weight_param
                    
                    logger.log_metrics(metrics, log_params)
                    
                    if recall_10 > best_recall:
                        best_recall = recall_10
                        best_params = log_params.copy()
                        best_model = model
    
    # Combined EASE grid search (Additive approach)
    logger.log("\n--- Combined EASE (Additive) ---")
    
    alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
    # Use best time params from previous experiments
    time_configs = [
        {'weight_mode': 'exponential', 'tau': 10.0, 'time_threshold': 5.0},
        {'weight_mode': 'binary', 'tau': 10.0, 'time_threshold': 5.0},
    ]
    
    for time_config in time_configs:
        for alpha in alphas:
            logger.log(f"\nBuilding combined matrix (alpha={alpha}, mode={time_config['weight_mode']})...")
            
            C_combined = builder.build_combined_cooccurrence_matrix(
                train_matrix=train_matrix,
                alpha=alpha,
                scale=None,  # Auto-scale
                max_time_diff=60.0,
                weight_mode=time_config['weight_mode'],
                tau=time_config['tau'],
                time_threshold=time_config['time_threshold'],
                max_window_size=50,
                verbose=False
            )
            
            for reg_weight in reg_weights:
                logger.log(f"  Trying λ={reg_weight}...")
                
                model = WeightedEASE(reg_weight=reg_weight)
                model.fit_with_cooccurrence(train_matrix, C_combined, verbose=False)
                
                predictions, _ = model.recommend(train_matrix, top_k=10, filter_already_liked=True)
                metrics = evaluate_all(predictions, valid_gt, k_values=[10])
                
                recall_10 = metrics['Recall@10']
                logger.log(f"    Recall@10: {recall_10:.4f}")
                
                log_params = {
                    'model': 'CombinedEASE',
                    'reg_weight': reg_weight,
                    'alpha': alpha,
                    'weight_mode': time_config['weight_mode'],
                }
                if time_config['weight_mode'] == 'exponential':
                    log_params['tau'] = time_config['tau']
                else:
                    log_params['time_threshold'] = time_config['time_threshold']
                
                logger.log_metrics(metrics, log_params)
                
                if recall_10 > best_recall:
                    best_recall = recall_10
                    best_params = log_params.copy()
                    best_model = model
    
    logger.log("\n" + "="*50)
    logger.log("Grid Search Complete!")
    logger.log(f"Best Recall@10: {best_recall:.4f}")
    logger.log(f"Best Parameters: {best_params}")
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
    elif args.combined:
        experiment_name = "combined_ease"
    elif args.weighted:
        experiment_name = "weighted_ease"
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
    elif args.combined:
        model, metrics = train_combined_ease(
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