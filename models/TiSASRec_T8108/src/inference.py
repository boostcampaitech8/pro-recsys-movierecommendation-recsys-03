"""
TiSASRec 추론 스크립트
학습된 모델을 사용하여 추천 결과 생성 및 submission 파일 생성

사용법:
    python inference.py
    python inference.py --checkpoint output/checkpoints/TiSASRec-ml_movie.pth
    python inference.py --topk 10
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple

# 현재 디렉터리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# RecBole imports
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_seed, get_model

# 로컬 TiSASRec 모델
from models.tisasrec import TiSASRec


def load_model(checkpoint_path: str, config: Config, dataset) -> TiSASRec:
    """
    저장된 체크포인트에서 모델 로드
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        config: RecBole Config 객체
        dataset: RecBole Dataset 객체
    
    Returns:
        로드된 TiSASRec 모델
    """
    model = TiSASRec(config, dataset).to(config['device'])
    
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model


def get_user_history(dataset, user_id: int) -> List[int]:
    """
    특정 유저의 상호작용 히스토리 조회
    
    Args:
        dataset: RecBole Dataset 객체
        user_id: 유저 ID (내부 인덱스)
    
    Returns:
        아이템 ID 리스트
    """
    # RecBole의 dataset에서 유저별 시퀀스 추출
    # 이 부분은 RecBole 버전에 따라 다를 수 있음
    return dataset.inter_feat[dataset.uid_field] == user_id


def generate_recommendations(
    model: TiSASRec,
    dataset,
    config: Config,
    topk: int = 10
) -> List[Tuple[int, List[int]]]:
    """
    모든 유저에 대해 Top-K 추천 생성
    
    Args:
        model: 학습된 TiSASRec 모델
        dataset: RecBole Dataset 객체
        config: RecBole Config 객체
        topk: 추천할 아이템 수
    
    Returns:
        (user_id, [item_ids]) 튜플 리스트
    """
    model.eval()
    
    recommendations = []
    
    # 유저별 상호작용 데이터 준비
    user_ids = dataset.inter_feat[dataset.uid_field].unique()
    
    print(f"Generating recommendations for {len(user_ids)} users...")
    
    with torch.no_grad():
        for user_id in tqdm(user_ids, desc="Generating recommendations"):
            # 해당 유저의 상호작용 시퀀스 가져오기
            user_inter = dataset.inter_feat[dataset.inter_feat[dataset.uid_field] == user_id]
            
            # 시간순 정렬
            if config['TIME_FIELD'] in user_inter.columns:
                user_inter = user_inter.sort_values(config['TIME_FIELD'])
            
            # 아이템 시퀀스 추출
            item_seq = user_inter[dataset.iid_field].values
            
            # 이미 상호작용한 아이템 (필터링용)
            interacted_items = set(item_seq)
            
            # 시퀀스 길이 제한 및 패딩
            max_len = config['MAX_ITEM_LIST_LENGTH']
            if len(item_seq) > max_len:
                item_seq = item_seq[-max_len:]
            
            # 입력 텐서 준비
            item_seq_tensor = torch.tensor(item_seq, dtype=torch.long).unsqueeze(0).to(config['device'])
            item_seq_len = torch.tensor([len(item_seq)], dtype=torch.long).to(config['device'])
            
            # 타임스탬프 시퀀스 (TiSASRec에서 필요)
            if config['TIME_FIELD'] in user_inter.columns:
                time_seq = user_inter[config['TIME_FIELD']].values
                if len(time_seq) > max_len:
                    time_seq = time_seq[-max_len:]
                time_seq_tensor = torch.tensor(time_seq, dtype=torch.long).unsqueeze(0).to(config['device'])
            
            # 모델 예측
            # TiSASRec의 full_sort_predict 또는 predict 메서드 사용
            try:
                # 전체 아이템에 대한 점수 계산
                scores = model.full_sort_predict(Interaction({
                    dataset.uid_field: torch.tensor([user_id], dtype=torch.long).to(config['device']),
                    'item_seq': item_seq_tensor,
                    'item_seq_len': item_seq_len
                }))
                scores = scores.cpu().numpy().flatten()
            except Exception as e:
                # 대체 방법: forward pass 직접 호출
                seq_output = model.forward(item_seq_tensor, item_seq_len)
                scores = torch.matmul(seq_output[:, -1, :], model.item_embedding.weight.T)
                scores = scores.cpu().numpy().flatten()
            
            # 이미 상호작용한 아이템 점수를 -inf로 설정
            for item in interacted_items:
                if item < len(scores):
                    scores[item] = -np.inf
            
            # 패딩 아이템(0) 제외
            scores[0] = -np.inf
            
            # Top-K 아이템 선택
            top_items = np.argsort(scores)[-topk:][::-1]
            
            recommendations.append((user_id, top_items.tolist()))
    
    return recommendations


def create_submission(
    recommendations: List[Tuple[int, List[int]]],
    dataset,
    original_data_path: str,
    output_path: str
) -> None:
    """
    submission.csv 파일 생성
    
    Args:
        recommendations: (user_id, [item_ids]) 튜플 리스트
        dataset: RecBole Dataset 객체
        original_data_path: 원본 train_ratings.csv 경로 (유저 ID 매핑용)
        output_path: submission 파일 저장 경로
    """
    print(f"Creating submission file...")
    
    # 원본 유저 ID 매핑 (RecBole 내부 인덱스 -> 원본 ID)
    # RecBole은 내부적으로 ID를 재매핑하므로 역매핑 필요
    
    # id2token 매핑 사용
    user_id2token = dataset.field2id_token[dataset.uid_field]
    item_id2token = dataset.field2id_token[dataset.iid_field]
    
    result = []
    for user_internal_id, item_internal_ids in recommendations:
        # 내부 ID를 원본 ID로 변환
        original_user_id = user_id2token[user_internal_id]
        
        for item_internal_id in item_internal_ids:
            original_item_id = item_id2token[item_internal_id]
            result.append((original_user_id, original_item_id))
    
    # DataFrame 생성 및 저장
    submission_df = pd.DataFrame(result, columns=['user', 'item'])
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    print(f"Total recommendations: {len(submission_df)}")
    print(f"Unique users: {submission_df['user'].nunique()}")
    
    # 샘플 출력
    print("\nSample recommendations:")
    print(submission_df.head(20))


def main():
    parser = argparse.ArgumentParser(description="Generate recommendations using trained TiSASRec")
    
    parser.add_argument('--config', type=str, default='../config/tisasrec.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='ml_movie',
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='../dataset/',
                        help='Path to dataset directory')
    parser.add_argument('--original_data', type=str, default='../../../data/train/train_ratings.csv',
                        help='Path to original train_ratings.csv')
    parser.add_argument('--output', type=str, default='../output/submission.csv',
                        help='Output submission file path')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of items to recommend per user')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    # Configuration
    parameter_dict = {
        'model': 'TiSASRec',
        'dataset': args.dataset,
        'data_path': args.data_path,
        'gpu_id': args.gpu_id,
        'eval_args': {
            'split': {'RS': [1.0, 0.0, 0.0]},  # 추론시에는 전체 데이터 사용
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'
        }
    }
    
    if os.path.exists(args.config):
        config = Config(
            model='TiSASRec',
            dataset=args.dataset,
            config_file_list=[args.config],
            config_dict=parameter_dict
        )
    else:
        config = Config(
            model='TiSASRec',
            dataset=args.dataset,
            config_dict=parameter_dict
        )
    
    # Initialize
    init_seed(config['seed'], config['reproducibility'])
    
    # Dataset
    print("Loading dataset...")
    dataset = create_dataset(config)
    
    # Find checkpoint if not specified
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # 기본 체크포인트 경로에서 찾기
        checkpoint_dir = config.get('checkpoint_dir', '../output/checkpoints/')
        possible_checkpoints = [
            os.path.join(checkpoint_dir, f'TiSASRec-{args.dataset}.pth'),
            os.path.join(checkpoint_dir, 'TiSASRec.pth'),
        ]
        
        for cp in possible_checkpoints:
            if os.path.exists(cp):
                checkpoint_path = cp
                break
        
        if checkpoint_path is None:
            print("Error: No checkpoint found. Please specify --checkpoint")
            sys.exit(1)
    
    # Load model
    model = load_model(checkpoint_path, config, dataset)
    
    # Generate recommendations
    recommendations = generate_recommendations(
        model=model,
        dataset=dataset,
        config=config,
        topk=args.topk
    )
    
    # Create submission
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_submission(
        recommendations=recommendations,
        dataset=dataset,
        original_data_path=args.original_data,
        output_path=args.output
    )
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
