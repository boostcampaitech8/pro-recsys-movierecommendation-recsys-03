"""
원본 MovieLens 데이터를 RecBole 형식으로 변환하는 스크립트

RecBole 데이터 형식:
- .inter 파일: user_id:token, item_id:token, timestamp:float 등의 컬럼
- .item 파일 (optional): item_id:token, genre:token_seq 등의 컬럼
"""

import os
import pandas as pd
import json
from pathlib import Path


def convert_ratings_to_inter(
    input_path: str,
    output_path: str,
    user_col: str = "user",
    item_col: str = "item",
    time_col: str = "time"
):
    """
    train_ratings.csv를 RecBole .inter 형식으로 변환
    
    Args:
        input_path: 원본 train_ratings.csv 경로
        output_path: 출력 .inter 파일 경로
        user_col: 유저 컬럼명
        item_col: 아이템 컬럼명
        time_col: 타임스탬프 컬럼명
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of users: {df[user_col].nunique()}")
    print(f"Number of items: {df[item_col].nunique()}")
    
    # RecBole 형식으로 컬럼명 변경
    # RecBole은 컬럼명:타입 형식의 헤더를 사용
    recbole_df = pd.DataFrame({
        "user_id:token": df[user_col],
        "item_id:token": df[item_col],
        "timestamp:float": df[time_col]
    })
    
    # .inter 파일로 저장 (탭 구분)
    recbole_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved RecBole .inter file to {output_path}")
    print(f"Total interactions: {len(recbole_df)}")
    
    return recbole_df


def convert_genres_to_item(
    genres_path: str,
    output_path: str
):
    """
    genres.tsv를 RecBole .item 형식으로 변환 (optional)
    
    Args:
        genres_path: 원본 genres.tsv 경로
        output_path: 출력 .item 파일 경로
    """
    print(f"Loading genres from {genres_path}...")
    genres_df = pd.read_csv(genres_path, sep="\t")
    
    print(f"Genres data shape: {genres_df.shape}")
    
    # item별 genre 리스트로 집계
    item_genres = genres_df.groupby("item")["genre"].apply(list).reset_index()
    
    # RecBole token_seq 형식으로 변환 (공백으로 구분)
    recbole_item_df = pd.DataFrame({
        "item_id:token": item_genres["item"],
        "genre:token_seq": item_genres["genre"].apply(lambda x: " ".join(x))
    })
    
    recbole_item_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved RecBole .item file to {output_path}")
    print(f"Total items with genres: {len(recbole_item_df)}")
    
    return recbole_item_df


def convert_years_to_item(
    years_path: str,
    item_df: pd.DataFrame = None,
    output_path: str = None
):
    """
    years.tsv 정보를 .item 파일에 추가
    
    Args:
        years_path: 원본 years.tsv 경로
        item_df: 기존 item DataFrame (genre 정보 포함)
        output_path: 출력 .item 파일 경로
    """
    print(f"Loading years from {years_path}...")
    years_df = pd.read_csv(years_path, sep="\t")
    
    if item_df is not None:
        # 기존 item_df와 병합
        merged = item_df.merge(
            years_df.rename(columns={"item": "item_id:token", "year": "year:float"}),
            on="item_id:token",
            how="left"
        )
        merged.to_csv(output_path, sep="\t", index=False)
        print(f"Updated .item file with year information")
        return merged
    else:
        recbole_years_df = pd.DataFrame({
            "item_id:token": years_df["item"],
            "year:float": years_df["year"]
        })
        recbole_years_df.to_csv(output_path, sep="\t", index=False)
        return recbole_years_df


def main():
    # 경로 설정
    # 실제 환경에 맞게 수정 필요
    data_dir = Path("../../data/train")
    output_dir = Path("./data/movielens")
    
    # 출력 디렉터리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. train_ratings.csv -> movielens.inter
    inter_output = output_dir / "movielens.inter"
    if (data_dir / "train_ratings.csv").exists():
        convert_ratings_to_inter(
            input_path=str(data_dir / "train_ratings.csv"),
            output_path=str(inter_output)
        )
    else:
        print(f"Warning: {data_dir / 'train_ratings.csv'} not found")
        print("Please check the data directory path")
    
    # 2. genres.tsv -> movielens.item (optional)
    item_output = output_dir / "movielens.item"
    if (data_dir / "genres.tsv").exists():
        item_df = convert_genres_to_item(
            genres_path=str(data_dir / "genres.tsv"),
            output_path=str(item_output)
        )
        
        # 3. years.tsv 정보 추가 (optional)
        if (data_dir / "years.tsv").exists():
            convert_years_to_item(
                years_path=str(data_dir / "years.tsv"),
                item_df=item_df,
                output_path=str(item_output)
            )
    
    print("\n" + "="*50)
    print("Data conversion completed!")
    print(f"Output directory: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
