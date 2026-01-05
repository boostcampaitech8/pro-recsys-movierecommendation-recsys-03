"""
데이터 변환 스크립트
대회 데이터(train_ratings.csv)를 RecBole atomic file 형식(.inter)으로 변환

RecBole atomic file 형식:
- 탭으로 구분된 파일
- 첫 번째 줄: 컬럼명:타입 형식의 헤더
- user_id:token, item_id:token, timestamp:float 형식 사용

사용법:
    python data_converter.py --input ../../../data/train/train_ratings.csv --output ../dataset/ml_movie
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def convert_ratings_to_inter(input_path: str, output_dir: str, dataset_name: str = "ml_movie") -> None:
    """
    train_ratings.csv를 RecBole .inter 형식으로 변환
    
    Args:
        input_path: train_ratings.csv 경로
        output_dir: 출력 디렉터리 경로
        dataset_name: 데이터셋 이름 (출력 파일명에 사용)
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 컬럼명 확인
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Total interactions: {len(df):,}")
    print(f"Unique users: {df['user'].nunique():,}")
    print(f"Unique items: {df['item'].nunique():,}")
    
    # RecBole 형식으로 컬럼명 변경
    # RecBole은 user_id:token, item_id:token, timestamp:float 형식을 사용
    df_recbole = df.rename(columns={
        'user': 'user_id',
        'item': 'item_id',
        'time': 'timestamp'
    })
    
    # 출력 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # .inter 파일 저장 (RecBole atomic file 형식)
    inter_file = os.path.join(output_dir, f"{dataset_name}.inter")
    
    # 헤더 작성: 컬럼명:타입 형식
    header = "user_id:token\titem_id:token\ttimestamp:float"
    
    # 데이터를 탭으로 구분하여 저장
    with open(inter_file, 'w') as f:
        f.write(header + '\n')
        for _, row in df_recbole.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\t{row['timestamp']}\n")
    
    print(f"Saved to {inter_file}")
    
    # 데이터 통계 출력
    print_statistics(df_recbole)


def convert_item_features(data_dir: str, output_dir: str, dataset_name: str = "ml_movie") -> None:
    """
    아이템 부가 정보(genres, years, directors, writers)를 RecBole .item 형식으로 변환
    (선택적 - TiSASRec 기본 모델에서는 사용하지 않음)
    
    Args:
        data_dir: 원본 데이터 디렉터리 (genres.tsv, years.tsv 등이 있는 곳)
        output_dir: 출력 디렉터리 경로
        dataset_name: 데이터셋 이름
    """
    print("\nConverting item features...")
    
    # years.tsv 로드
    years_path = os.path.join(data_dir, "years.tsv")
    if os.path.exists(years_path):
        years_df = pd.read_csv(years_path, sep='\t')
        print(f"Loaded years: {len(years_df)} items")
    else:
        print(f"Warning: {years_path} not found")
        years_df = None
    
    # genres.tsv 로드 (한 아이템에 여러 장르 가능)
    genres_path = os.path.join(data_dir, "genres.tsv")
    if os.path.exists(genres_path):
        genres_df = pd.read_csv(genres_path, sep='\t')
        # 아이템별로 장르를 리스트로 묶기
        genres_grouped = genres_df.groupby('item')['genre'].apply(list).reset_index()
        genres_grouped['genre'] = genres_grouped['genre'].apply(lambda x: ' '.join(x))
        print(f"Loaded genres: {len(genres_grouped)} items")
    else:
        print(f"Warning: {genres_path} not found")
        genres_grouped = None
    
    # 아이템 피처 병합
    if years_df is not None:
        item_df = years_df.rename(columns={'item': 'item_id', 'year': 'year'})
        
        if genres_grouped is not None:
            genres_grouped = genres_grouped.rename(columns={'item': 'item_id'})
            item_df = item_df.merge(genres_grouped, on='item_id', how='left')
            item_df['genre'] = item_df['genre'].fillna('')
        
        # .item 파일 저장
        item_file = os.path.join(output_dir, f"{dataset_name}.item")
        
        # 헤더 작성
        if genres_grouped is not None:
            header = "item_id:token\tyear:float\tgenre:token_seq"
            with open(item_file, 'w') as f:
                f.write(header + '\n')
                for _, row in item_df.iterrows():
                    f.write(f"{row['item_id']}\t{row['year']}\t{row['genre']}\n")
        else:
            header = "item_id:token\tyear:float"
            with open(item_file, 'w') as f:
                f.write(header + '\n')
                for _, row in item_df.iterrows():
                    f.write(f"{row['item_id']}\t{row['year']}\n")
        
        print(f"Saved to {item_file}")


def print_statistics(df: pd.DataFrame) -> None:
    """데이터셋 통계 출력"""
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    print(f"Total interactions: {len(df):,}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique items: {df['item_id'].nunique():,}")
    print(f"Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.6f}")
    
    # 시간 범위
    min_time = pd.to_datetime(df['timestamp'].min(), unit='s')
    max_time = pd.to_datetime(df['timestamp'].max(), unit='s')
    print(f"Time range: {min_time} ~ {max_time}")
    
    # 유저별 상호작용 수 분포
    user_counts = df.groupby('user_id').size()
    print(f"\nInteractions per user:")
    print(f"  Min: {user_counts.min()}")
    print(f"  Max: {user_counts.max()}")
    print(f"  Mean: {user_counts.mean():.2f}")
    print(f"  Median: {user_counts.median():.2f}")
    
    # 아이템별 상호작용 수 분포
    item_counts = df.groupby('item_id').size()
    print(f"\nInteractions per item:")
    print(f"  Min: {item_counts.min()}")
    print(f"  Max: {item_counts.max()}")
    print(f"  Mean: {item_counts.mean():.2f}")
    print(f"  Median: {item_counts.median():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Convert competition data to RecBole format")
    parser.add_argument(
        "--input", 
        type=str, 
        default="../../../data/train/train_ratings.csv",
        help="Path to train_ratings.csv"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../data/train",
        help="Path to data directory containing genres.tsv, years.tsv, etc."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="../dataset/ml_movie",
        help="Output directory for RecBole atomic files"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ml_movie",
        help="Dataset name for output files"
    )
    parser.add_argument(
        "--include_item_features",
        action="store_true",
        help="Include item features (genres, years) in conversion"
    )
    
    args = parser.parse_args()
    
    # 메인 상호작용 데이터 변환
    convert_ratings_to_inter(args.input, args.output, args.dataset_name)
    
    # 아이템 피처 변환 (선택적)
    if args.include_item_features:
        convert_item_features(args.data_dir, args.output, args.dataset_name)
    
    print("\nConversion completed!")
    print(f"Output files are in: {args.output}")


if __name__ == "__main__":
    main()
