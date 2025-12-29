#!/usr/bin/env python
"""
Inference script for EASE model.
Generates submission file for competition.

Usage:
    # Basic EASE
    python run_inference.py --reg_weight 500
    
    # Session-based Weighted EASE
    python run_inference.py --weighted --session_threshold 1800 --page_threshold 30 --alpha 0.3
    
    # Load pre-trained model
    python run_inference.py --model_path ./output/ease_model.npy
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
from utils import set_seed, ensure_dir, create_submission


def parse_args():
    parser = argparse.ArgumentParser(description='EASE model inference')
    
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
    
    # Pre-trained model
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model weights')
    
    # Recommendation parameters
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of items to recommend per user')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    
    print("="*50)
    print("EASE Model Inference")
    print("="*50)
    
    # Load data (full dataset, no holdout)
    print("\nLoading data...")
    data_file = os.path.join(args.data_dir, 'train_ratings.csv')
    data_loader = DataLoader(data_file, seed=args.seed)
    
    # Create full interaction matrix
    full_matrix = data_loader.create_submission_matrix()
    print(f"Full matrix: {full_matrix.shape}, nnz={full_matrix.nnz}")
    
    # Load or train model
    if args.model_path and os.path.exists(args.model_path):
        print(f"\nLoading model from: {args.model_path}")
        B = np.load(args.model_path)
        
        if args.weighted:
            model = WeightedEASE(reg_weight=args.reg_weight)
        else:
            model = EASE(reg_weight=args.reg_weight)
        
        model.B = B
        model.n_items = B.shape[0]
    else:
        print(f"\nTraining model from scratch...")
        
        if args.weighted:
            print(f"  Session threshold: {args.session_threshold}s ({args.session_threshold/60:.1f} min)")
            print(f"  Page threshold: {args.page_threshold}s")
            print(f"  Within-page weight: {args.within_page_weight}")
            print(f"  Cross-page tau: {args.cross_page_tau}s")
            print(f"  Alpha: {args.alpha}")
            
            builder = WeightedMatrixBuilder(data_loader)
            C_combined = builder.build_combined_cooccurrence_matrix(
                train_matrix=full_matrix,
                alpha=args.alpha,
                session_threshold=args.session_threshold,
                page_threshold=args.page_threshold,
                within_page_weight=args.within_page_weight,
                cross_page_tau=args.cross_page_tau,
                verbose=True
            )
            
            model = WeightedEASE(reg_weight=args.reg_weight)
            model.fit_with_cooccurrence(full_matrix, C_combined, verbose=True)
        else:
            model = EASE(reg_weight=args.reg_weight)
            model.fit(full_matrix, verbose=True)
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    predictions, scores = model.recommend(
        full_matrix,
        top_k=args.top_k,
        filter_already_liked=True
    )
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Create submission file
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    create_submission(
        predictions=predictions,
        idx2user=data_loader.idx2user,
        idx2item=data_loader.idx2item,
        output_path=submission_path,
        top_k=args.top_k
    )
    
    # Save model if trained from scratch
    if args.model_path is None or not os.path.exists(args.model_path):
        model_path = os.path.join(args.output_dir, 'ease_model.npy')
        np.save(model_path, model.B)
        print(f"Model saved to: {model_path}")
    
    print("\n" + "="*50)
    print("Inference complete!")
    print(f"Submission file: {submission_path}")
    print("="*50)


if __name__ == "__main__":
    main()