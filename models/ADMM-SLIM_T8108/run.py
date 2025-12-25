"""
ADMM-SLIM 모델 학습 실행 스크립트

RecBole 라이브러리를 활용하여 ADMMSLIM 모델을 학습합니다.
"""

import argparse
import os
import sys
from pathlib import Path

from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer
from recbole.model.general_recommender import ADMMSLIM

import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run ADMMSLIM model")
    
    # 기본 설정
    parser.add_argument(
        "--config_files",
        type=str,
        nargs="+",
        default=["config/dataset.yaml", "config/model.yaml"],
        help="Config file paths"
    )
    
    # 하이퍼파라미터 오버라이드
    parser.add_argument("--lambda1", type=float, default=None, help="L1 regularization")
    parser.add_argument("--lambda2", type=float, default=None, help="L2 regularization")
    parser.add_argument("--alpha", type=float, default=None, help="Popularity weight")
    parser.add_argument("--rho", type=float, default=None, help="ADMM convergence param")
    parser.add_argument("--k", type=int, default=None, help="ADMM iterations")
    
    # 실험 설정
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    
    return parser.parse_args()


def build_config(args):
    """
    Config 객체 생성
    """
    # 파라미터 딕셔너리 구성
    parameter_dict = {
        "seed": args.seed,
        "gpu_id": args.gpu_id,
    }
    
    # 하이퍼파라미터 오버라이드
    if args.lambda1 is not None:
        parameter_dict["lambda1"] = args.lambda1
    if args.lambda2 is not None:
        parameter_dict["lambda2"] = args.lambda2
    if args.alpha is not None:
        parameter_dict["alpha"] = args.alpha
    if args.rho is not None:
        parameter_dict["rho"] = args.rho
    if args.k is not None:
        parameter_dict["k"] = args.k
    
    # Config 생성
    config = Config(
        model="ADMMSLIM",
        dataset="movielens",
        config_file_list=args.config_files,
        config_dict=parameter_dict
    )
    
    return config


def run_experiment(args):
    """
    실험 실행
    """
    print("="*60)
    print("ADMM-SLIM Experiment")
    print("="*60)
    
    # Config 생성
    config = build_config(args)
    
    # 시드 초기화
    init_seed(config["seed"], config["reproducibility"])
    
    # 로거 초기화
    init_logger(config)
    logger = logging.getLogger()
    
    logger.info(f"Config: {config}")
    
    # 데이터셋 생성
    logger.info("Creating dataset...")
    dataset = create_dataset(config)
    logger.info(f"Dataset: {dataset}")
    
    # 데이터 준비 (train/valid/test 분할)
    logger.info("Preparing data...")
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 모델 생성
    logger.info("Creating ADMMSLIM model...")
    model = ADMMSLIM(config, train_data.dataset)
    logger.info(f"Model: {model}")
    
    # Trainer 생성
    trainer = Trainer(config, model)
    
    # 학습 (ADMMSLIM은 fit 과정에서 closed-form solution 계산)
    logger.info("Training model...")
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )
    
    logger.info(f"Best valid score: {best_valid_score}")
    logger.info(f"Best valid result: {best_valid_result}")
    
    # 테스트
    # Note: load_best_model=False로 설정 (PyTorch 2.6+ 호환성 문제 우회)
    # ADMMSLIM은 __init__에서 학습이 완료되므로 체크포인트 로드 불필요
    logger.info("Evaluating on test set...")
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    logger.info(f"Test result: {test_result}")
    
    print("\n" + "="*60)
    print("Experiment Results")
    print("="*60)
    print(f"Valid Recall@10: {best_valid_result.get('recall@10', 'N/A')}")
    print(f"Test Recall@10: {test_result.get('recall@10', 'N/A')}")
    print("="*60)
    
    return {
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "model": model,
        "trainer": trainer,
        "config": config
    }


def run_simple():
    """
    간단하게 run_recbole 함수로 실행
    """
    result = run_recbole(
        model="ADMMSLIM",
        dataset="movielens",
        config_file_list=["config/dataset.yaml", "config/model.yaml"]
    )
    return result


if __name__ == "__main__":
    args = parse_args()
    
    # 작업 디렉터리 설정
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 실험 실행
    results = run_experiment(args)