"""
ADMM-SLIM 추론 및 제출 파일 생성 스크립트

전체 학습 데이터를 사용하여 모델을 학습하고,
각 유저에 대한 Top-10 추천을 생성하여 제출 파일을 만듭니다.

RecBole ADMMSLIM 소스코드 참고:
- interaction_matrix: 원본 user-item 상호작용 행렬 (sparse)
- item_similarity: 학습된 item-item 유사도 행렬 B (sparse)
- full_sort_predict: user에 대한 전체 아이템 스코어 계산
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from recbole.model.general_recommender import ADMMSLIM


def parse_args():
    parser = argparse.ArgumentParser(description="ADMMSLIM Inference")
    
    parser.add_argument(
        "--config_files",
        type=str,
        nargs="+",
        default=["config/dataset.yaml", "config/model.yaml"],
        help="Config file paths"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/submission.csv",
        help="Output submission file path"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of items to recommend per user"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for inference"
    )
    
    # 하이퍼파라미터 오버라이드
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--k", type=int, default=None)
    
    return parser.parse_args()


def get_user_item_mappings(dataset):
    """RecBole 내부 ID와 원본 ID 간의 매핑 정보 추출"""
    user_id2token = dataset.field2id_token["user_id"]
    item_id2token = dataset.field2id_token["item_id"]
    return user_id2token, item_id2token


def get_user_history_from_matrix(interaction_matrix, n_users):
    """
    상호작용 행렬에서 각 유저의 히스토리 추출
    
    Args:
        interaction_matrix: scipy sparse matrix (n_users, n_items)
        n_users: 유저 수
    
    Returns:
        dict: {user_id: set(item_ids)}
    """
    user_history = {}
    
    # CSR 형식으로 변환하여 효율적으로 접근
    csr_matrix = interaction_matrix.tocsr()
    
    for uid in range(n_users):
        # 해당 유저의 상호작용 아이템 인덱스
        start_idx = csr_matrix.indptr[uid]
        end_idx = csr_matrix.indptr[uid + 1]
        item_indices = csr_matrix.indices[start_idx:end_idx]
        user_history[uid] = set(item_indices.tolist())
    
    return user_history


def generate_recommendations(model, dataset, topk=10, batch_size=512):
    """
    전체 유저에 대한 Top-K 추천 생성
    
    ADMMSLIM 스코어 계산:
    - score = interaction_matrix[user] @ item_similarity
    - 이미 본 아이템은 제외
    """
    model.eval()
    
    # ID 매핑
    user_id2token, item_id2token = get_user_item_mappings(dataset)
    
    n_users = dataset.user_num
    n_items = dataset.item_num
    
    print(f"Number of users (including padding): {n_users}")
    print(f"Number of items (including padding): {n_items}")
    
    # 모델에서 필요한 행렬 가져오기
    # interaction_matrix: (n_users, n_items) - 원본 상호작용 행렬
    # item_similarity: (n_items, n_items) - 학습된 B 행렬
    interaction_matrix = model.interaction_matrix  # scipy sparse
    item_similarity = model.item_similarity  # scipy sparse
    
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"Item similarity matrix shape: {item_similarity.shape}")
    
    # 유저 히스토리 추출 (이미 본 아이템 제외용)
    user_history = get_user_history_from_matrix(interaction_matrix, n_users)
    
    recommendations = {}
    
    # 유저 ID 리스트 (0은 padding이므로 제외)
    user_ids = list(range(1, n_users))
    
    print(f"\nGenerating recommendations for {len(user_ids)} users...")
    
    for start_idx in tqdm(range(0, len(user_ids), batch_size)):
        end_idx = min(start_idx + batch_size, len(user_ids))
        batch_user_ids = user_ids[start_idx:end_idx]
        
        # 배치 유저의 상호작용 벡터 (sparse -> dense)
        batch_interactions = interaction_matrix[batch_user_ids, :].toarray()
        
        # 스코어 계산: score = X @ B
        # item_similarity가 sparse인 경우 처리
        if hasattr(item_similarity, 'toarray'):
            item_sim_dense = item_similarity.toarray()
        else:
            item_sim_dense = item_similarity
        
        batch_scores = batch_interactions @ item_sim_dense
        
        for i, uid in enumerate(batch_user_ids):
            scores = batch_scores[i].copy()
            
            # 이미 본 아이템 제외
            if uid in user_history:
                for seen_item in user_history[uid]:
                    scores[seen_item] = -np.inf
            
            # padding index (0) 제외
            scores[0] = -np.inf
            
            # Top-K 아이템 선택
            topk_indices = np.argsort(scores)[-topk:][::-1]
            
            # 원본 ID로 변환
            original_uid = user_id2token[uid]
            original_items = [item_id2token[iid] for iid in topk_indices]
            
            recommendations[original_uid] = original_items
    
    return recommendations


def save_submission(recommendations, output_path, topk=10):
    """대회 제출 형식으로 저장"""
    print(f"\nSaving submission to {output_path}...")
    
    rows = []
    for user_id, items in recommendations.items():
        for item_id in items[:topk]:
            rows.append({"user": user_id, "item": item_id})
    
    submission_df = pd.DataFrame(rows)
    
    # user 기준 정렬
    submission_df = submission_df.sort_values(by=["user"]).reset_index(drop=True)
    
    # 저장
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission saved!")
    print(f"Total rows: {len(submission_df)}")
    print(f"Number of users: {submission_df['user'].nunique()}")
    print(f"Items per user: {len(submission_df) // submission_df['user'].nunique()}")
    print(f"\nSample (first 20 rows):")
    print(submission_df.head(20))
    
    return submission_df


def main():
    args = parse_args()
    
    # 작업 디렉터리 설정
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("="*60)
    print("ADMM-SLIM Inference for Submission")
    print("="*60)
    
    # 파라미터 딕셔너리 구성
    parameter_dict = {
        # 전체 데이터를 학습에 사용 (분할 없음)
        "eval_args": {
            "split": {"RS": [1.0, 0.0, 0.0]},
            "mode": "full",
            "order": "RO",
        },
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
    
    # 시드 설정
    reproducibility = config["reproducibility"] if "reproducibility" in config.final_config_dict else True
    init_seed(config["seed"], reproducibility)
    
    # 데이터셋 생성 (전체 데이터)
    print("\nLoading full dataset...")
    dataset = create_dataset(config)
    print(f"Dataset: {dataset}")
    
    # 모델 생성 (ADMMSLIM은 __init__에서 학습 완료)
    print("\nTraining ADMMSLIM on full dataset...")
    model = ADMMSLIM(config, dataset)
    print("Model training completed!")
    
    # 모델 속성 확인
    print(f"\nModel attributes:")
    print(f"  - interaction_matrix type: {type(model.interaction_matrix)}")
    print(f"  - item_similarity type: {type(model.item_similarity)}")
    
    # 추천 생성
    recommendations = generate_recommendations(
        model=model,
        dataset=dataset,
        topk=args.topk,
        batch_size=args.batch_size
    )
    
    # 제출 파일 저장
    submission_df = save_submission(
        recommendations=recommendations,
        output_path=args.output,
        topk=args.topk
    )
    
    print("\n" + "="*60)
    print("Inference completed!")
    print(f"Submission file: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()