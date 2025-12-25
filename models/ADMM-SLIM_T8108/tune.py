"""
ADMM-SLIM 하이퍼파라미터 튜닝 스크립트

PyTorch 2.6+ 호환 버전
- run_recbole() 대신 직접 학습/평가 수행
- 체크포인트 로드 문제 우회
"""

import argparse
import os
import sys
from pathlib import Path
from itertools import product
import json
from datetime import datetime

import numpy as np
import pandas as pd

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from recbole.model.general_recommender import ADMMSLIM
from recbole.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="ADMMSLIM Hyperparameter Tuning")
    
    parser.add_argument(
        "--config_files",
        type=str,
        nargs="+",
        default=["config/dataset.yaml", "config/model.yaml"],
        help="Config file paths"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/tuning",
        help="Directory to save tuning results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--grid_size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Grid search size"
    )
    
    return parser.parse_args()


# 탐색할 하이퍼파라미터 공간
PARAM_GRIDS = {
    "small": {
        "lambda1": [1.0, 5.0, 10.0],
        "lambda2": [500.0, 1000.0],
        "alpha": [0.0, 0.5],
        "rho": [10000.0],
        "k": [100],
    },
    "medium": {
        "lambda1": [1.0, 3.0, 5.0, 10.0],
        "lambda2": [100.0, 500.0, 1000.0, 2000.0],
        "alpha": [0.0, 0.25, 0.5],
        "rho": [10000.0],
        "k": [100],
    },
    "large": {
        "lambda1": [1.0, 3.0, 5.0, 10.0, 20.0],
        "lambda2": [100.0, 500.0, 1000.0, 2000.0, 5000.0],
        "alpha": [0.0, 0.25, 0.5, 0.75],
        "rho": [1000.0, 10000.0, 100000.0],
        "k": [50, 100, 200],
    },
}


def run_single_experiment(config_files, params, seed=42):
    """
    단일 하이퍼파라미터 조합으로 실험 실행
    PyTorch 2.6+ 호환 버전
    """
    try:
        # Config 생성
        config = Config(
            model="ADMMSLIM",
            dataset="movielens",
            config_file_list=config_files,
            config_dict={
                **params,
                "seed": seed,
                "show_progress": False,
            }
        )
        
        # 시드 초기화
        init_seed(config["seed"], True)
        
        # 데이터셋 생성
        dataset = create_dataset(config)
        
        # 데이터 준비 (train/valid/test 분할)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        # 모델 생성 (ADMMSLIM은 __init__에서 학습 완료)
        model = ADMMSLIM(config, train_data.dataset)
        
        # Trainer 생성
        trainer = Trainer(config, model)
        
        # Valid 평가 (체크포인트 로드 없이)
        valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=False)
        
        # Test 평가
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        
        return {
            "params": params,
            "valid_recall@10": valid_result.get("recall@10", 0),
            "valid_ndcg@10": valid_result.get("ndcg@10", 0),
            "test_recall@10": test_result.get("recall@10", 0),
            "test_ndcg@10": test_result.get("ndcg@10", 0),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error with params {params}: {str(e)}")
        return {
            "params": params,
            "valid_recall@10": 0,
            "valid_ndcg@10": 0,
            "test_recall@10": 0,
            "test_ndcg@10": 0,
            "status": "failed",
            "error": str(e)
        }


def grid_search(config_files, param_grid, output_dir, seed=42):
    """
    Grid Search로 하이퍼파라미터 탐색
    """
    # 모든 조합 생성
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    print(f"Total combinations to try: {len(all_combinations)}")
    print(f"Parameters: {param_names}")
    print()
    
    results = []
    best_score = 0
    best_params = None
    
    for i, values in enumerate(all_combinations):
        params = dict(zip(param_names, values))
        
        print(f"[{i+1}/{len(all_combinations)}] Testing: {params}")
        
        result = run_single_experiment(config_files, params, seed)
        results.append(result)
        
        # 결과 출력
        if result["status"] == "success":
            print(f"  -> Valid Recall@10: {result['valid_recall@10']:.4f}, NDCG@10: {result['valid_ndcg@10']:.4f}")
            
            # 최고 성능 업데이트
            if result["valid_recall@10"] > best_score:
                best_score = result["valid_recall@10"]
                best_params = params
                print(f"  -> New best!")
        else:
            print(f"  -> Failed: {result.get('error', 'Unknown error')[:50]}...")
        
        print()
        
        # 중간 결과 저장
        save_results(results, output_dir)
    
    return results, best_params, best_score


def save_results(results, output_dir):
    """
    튜닝 결과 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # DataFrame으로 변환
    records = []
    for r in results:
        record = {
            **r["params"],
            "valid_recall@10": r["valid_recall@10"],
            "valid_ndcg@10": r["valid_ndcg@10"],
            "test_recall@10": r["test_recall@10"],
            "test_ndcg@10": r["test_ndcg@10"],
            "status": r["status"]
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # CSV 저장 (덮어쓰기)
    csv_path = os.path.join(output_dir, "tuning_results.csv")
    df.to_csv(csv_path, index=False)
    
    return df


def analyze_results(results_df):
    """
    튜닝 결과 분석
    """
    # 성공한 결과만 필터링
    success_df = results_df[results_df["status"] == "success"]
    
    if len(success_df) == 0:
        print("No successful experiments!")
        return None
    
    print("\n" + "="*60)
    print("Tuning Results Analysis")
    print("="*60)
    
    # 각 파라미터별 영향 분석
    param_cols = ["lambda1", "lambda2", "alpha", "rho", "k"]
    
    for param in param_cols:
        if param in success_df.columns and success_df[param].nunique() > 1:
            print(f"\n{param} impact on Valid Recall@10:")
            grouped = success_df.groupby(param)["valid_recall@10"].agg(["mean", "std", "max"])
            print(grouped.round(4))
    
    # Top 5 configurations
    print("\n" + "-"*60)
    print("Top 5 configurations (by Valid Recall@10):")
    print("-"*60)
    top5 = success_df.nlargest(5, "valid_recall@10")
    display_cols = ["lambda1", "lambda2", "alpha", "valid_recall@10", "valid_ndcg@10"]
    print(top5[display_cols].to_string(index=False))
    
    return top5


def main():
    args = parse_args()
    
    # 작업 디렉터리 설정
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("="*60)
    print("ADMM-SLIM Hyperparameter Tuning")
    print("="*60)
    
    # 그리드 선택
    param_grid = PARAM_GRIDS[args.grid_size]
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    
    print(f"Grid size: {args.grid_size}")
    print(f"Total combinations: {total_combinations}")
    print(f"Parameter grid: {param_grid}")
    print()
    
    # Grid Search 실행
    results, best_params, best_score = grid_search(
        config_files=args.config_files,
        param_grid=param_grid,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # 최종 결과 저장
    results_df = save_results(results, args.output_dir)
    
    # 결과 분석
    analyze_results(results_df)
    
    print("\n" + "="*60)
    print("Tuning completed!")
    print("="*60)
    if best_params:
        print(f"Best parameters: {best_params}")
        print(f"Best Valid Recall@10: {best_score:.4f}")
    else:
        print("No successful experiments.")
    print(f"\nResults saved to: {args.output_dir}/tuning_results.csv")
    print("="*60)


if __name__ == "__main__":
    main()