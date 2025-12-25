"""
ADMM-SLIM 프로젝트 유틸리티 함수

공통적으로 사용되는 함수들을 모아놓은 모듈입니다.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """
    재현성을 위한 시드 설정
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_train_ratings(data_path: str) -> pd.DataFrame:
    """
    train_ratings.csv 로드
    """
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} interactions")
    print(f"Users: {df['user'].nunique()}, Items: {df['item'].nunique()}")
    return df


def compute_recall_at_k(predictions: dict, ground_truth: dict, k: int = 10) -> float:
    """
    Recall@K 계산
    
    Args:
        predictions: {user_id: [predicted_items]} 형태
        ground_truth: {user_id: [actual_items]} 형태
        k: top-k
    
    Returns:
        recall@k 점수
    """
    recalls = []
    
    for user_id in ground_truth:
        if user_id not in predictions:
            recalls.append(0.0)
            continue
        
        pred_items = set(predictions[user_id][:k])
        true_items = set(ground_truth[user_id])
        
        if len(true_items) == 0:
            continue
        
        hit = len(pred_items & true_items)
        recall = hit / len(true_items)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def compute_ndcg_at_k(predictions: dict, ground_truth: dict, k: int = 10) -> float:
    """
    NDCG@K 계산
    
    Args:
        predictions: {user_id: [predicted_items]} 형태 (순서 중요)
        ground_truth: {user_id: [actual_items]} 형태
        k: top-k
    
    Returns:
        ndcg@k 점수
    """
    ndcgs = []
    
    for user_id in ground_truth:
        if user_id not in predictions:
            ndcgs.append(0.0)
            continue
        
        pred_items = predictions[user_id][:k]
        true_items = set(ground_truth[user_id])
        
        if len(true_items) == 0:
            continue
        
        # DCG 계산
        dcg = 0.0
        for i, item in enumerate(pred_items):
            if item in true_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because position is 1-indexed
        
        # IDCG 계산 (최적 순서)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true_items))))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """
    모델 체크포인트 저장
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path):
    """
    체크포인트 로드
    """
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return None
    
    checkpoint = torch.load(path)
    print(f"Loaded checkpoint from {path}")
    print(f"Epoch: {checkpoint.get('epoch')}, Metrics: {checkpoint.get('metrics')}")
    
    return checkpoint


def create_submission(predictions: dict, output_path: str, topk: int = 10):
    """
    대회 제출 형식의 CSV 파일 생성
    
    Args:
        predictions: {user_id: [item_ids]} 형태
        output_path: 출력 파일 경로
        topk: 각 유저당 추천 아이템 수
    """
    rows = []
    for user_id, items in predictions.items():
        for item_id in items[:topk]:
            rows.append({"user": user_id, "item": item_id})
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["user"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    print(f"Total rows: {len(df)}, Users: {df['user'].nunique()}")
    
    return df


def analyze_data(ratings_df: pd.DataFrame):
    """
    데이터 기본 분석
    """
    print("="*60)
    print("Data Analysis")
    print("="*60)
    
    print(f"\nBasic Statistics:")
    print(f"  Total interactions: {len(ratings_df):,}")
    print(f"  Unique users: {ratings_df['user'].nunique():,}")
    print(f"  Unique items: {ratings_df['item'].nunique():,}")
    print(f"  Density: {len(ratings_df) / (ratings_df['user'].nunique() * ratings_df['item'].nunique()) * 100:.4f}%")
    
    # 유저당 상호작용 수
    user_counts = ratings_df.groupby("user").size()
    print(f"\nInteractions per user:")
    print(f"  Mean: {user_counts.mean():.2f}")
    print(f"  Median: {user_counts.median():.2f}")
    print(f"  Min: {user_counts.min()}")
    print(f"  Max: {user_counts.max()}")
    
    # 아이템당 상호작용 수
    item_counts = ratings_df.groupby("item").size()
    print(f"\nInteractions per item:")
    print(f"  Mean: {item_counts.mean():.2f}")
    print(f"  Median: {item_counts.median():.2f}")
    print(f"  Min: {item_counts.min()}")
    print(f"  Max: {item_counts.max()}")
    
    # 시간 범위
    if "time" in ratings_df.columns:
        from datetime import datetime
        min_time = datetime.fromtimestamp(ratings_df["time"].min())
        max_time = datetime.fromtimestamp(ratings_df["time"].max())
        print(f"\nTime range:")
        print(f"  From: {min_time}")
        print(f"  To: {max_time}")
    
    print("="*60)


class EarlyStopping:
    """
    Early Stopping 구현
    """
    def __init__(self, patience: int = 10, delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: 개선 없이 기다릴 epoch 수
            delta: 개선으로 인정할 최소 변화량
            mode: 'max' (높을수록 좋음) 또는 'min' (낮을수록 좋음)
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_timestamp():
    """현재 타임스탬프 반환"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    """디렉터리 존재 확인 및 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)
