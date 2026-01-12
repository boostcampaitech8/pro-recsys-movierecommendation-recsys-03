import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from utils import create_item_feature_matrix

class MovieDataset(Dataset):
    def __init__(self, train_matrix, user_side_matrices):
        self.train_matrix = train_matrix
        self.user_side_matrices = user_side_matrices # 이제 dict 형태의 Tensor들을 받음

    def __len__(self):
        return self.train_matrix.shape[0]

    def __getitem__(self, idx):
        # 유저의 상호작용 벡터
        user_vector = torch.FloatTensor(self.train_matrix[idx].toarray()).squeeze()
        
        # 미리 계산된 유저별 Side Info 벡터들 (연산 없이 바로 추출)
        user_side_dict = {
            key: matrix[idx] for key, matrix in self.user_side_matrices.items()
        }
            
        return user_vector, user_side_dict

def get_loaders(config):
    base_url = config['data']['base_url']
    
    # 데이터 로드
    train_df = pd.read_csv(base_url + 'train_ratings.csv')
    genre_df = pd.read_csv(base_url + 'genres.tsv', sep='\t')
    director_df = pd.read_csv(base_url + 'directors.tsv', sep='\t')
    year_df = pd.read_csv(base_url + 'years.tsv', sep='\t')
    writer_df = pd.read_csv(base_url + 'writers.tsv', sep='\t')

    year_df['year'] = (year_df['year'] // 20) * 20

    def group_rare_features(df, col_name, min_cnt):
        counts = df[col_name].value_counts()
        # 기준치 미만인 값들의 리스트
        rare_labels = counts[counts < min_cnt].index
        
        # 해당 값들을 모두 'RARE_XXX'로 치환
        df_copy = df.copy()
        df_copy.loc[df_copy[col_name].isin(rare_labels), col_name] = f'RARE_{col_name}'
        
        return df_copy
    
    print(f"전처리 전 감독 수: {director_df['director'].nunique()}")
    director_df = group_rare_features(director_df, 'director', min_cnt=5)
    print(f"그룹화 후 감독 수: {director_df['director'].nunique()} (RARE 그룹 포함)")

    print(f"전처리 전 작가 수: {writer_df['writer'].nunique()}")
    writer_df = group_rare_features(writer_df, 'writer', min_cnt=4) 
    print(f"그룹화 후 작가 수: {writer_df['writer'].nunique()} (RARE 그룹 포함)")

    '''
    def group_features_by_tier(df, col_name):
        counts = df[col_name].value_counts()
        
        # 등급 나누기 (제안하신 기준 반영)
        top_labels = counts[counts >= 30].index
        mid_labels = counts[(counts < 30) & (counts >= 10)].index
        bot_labels = counts[counts < 10].index
        
        df_copy = df.copy()
        
        # 각 레이블 치환
        df_copy.loc[df_copy[col_name].isin(top_labels), col_name] = f'TOP_{col_name}'
        df_copy.loc[df_copy[col_name].isin(mid_labels), col_name] = f'MID_{col_name}'
        df_copy.loc[df_copy[col_name].isin(bot_labels), col_name] = f'BOT_{col_name}'
        
        return df_copy

    # 적용 부분
    print(f"전처리 전 감독 수: {director_df['director'].nunique()}")
    director_df = group_features_by_tier(director_df, 'director')
    print(f"그룹화 후 감독 등급: {director_df['director'].unique()}") # 3개 등급만 남음

    print(f"전처리 전 작가 수: {writer_df['writer'].nunique()}")
    writer_df = group_features_by_tier(writer_df, 'writer')
    print(f"그룹화 후 작가 등급: {writer_df['writer'].unique()}") # 3개 등급만 남음'''

    # ID 리매핑: 모든 데이터의 ID 체계를 0부터 시작하는 인덱스로 통일
    all_item_ids = (set(train_df['item']) | set(genre_df['item']) | 
                    set(director_df['item']) | set(year_df['item']) | 
                    set(writer_df['item']))
    unique_items = sorted(all_item_ids)
    item2idx = {item: i for i, item in enumerate(unique_items)}
    n_items = len(unique_items)
    
    unique_users = sorted(train_df['user'].unique())
    user2idx = {user: i for i, user in enumerate(unique_users)}
    n_users = len(unique_users)

    # 데이터프레임 인덱스 업데이트
    train_df['item_idx'] = train_df['item'].map(item2idx)
    train_df['user_idx'] = train_df['user'].map(user2idx)
    genre_df['item_idx'] = genre_df['item'].map(item2idx)
    director_df['item_idx'] = director_df['item'].map(item2idx)
    year_df['item_idx'] = year_df['item'].map(item2idx)
    writer_df['item_idx'] = writer_df['item'].map(item2idx)

    # Side Info Matrices 생성 (utils.py 함수 사용)
    genre_matrix, genre_dim = create_item_feature_matrix(genre_df, 'item_idx', 'genre', n_items)
    director_matrix, director_dim = create_item_feature_matrix(director_df, 'item_idx', 'director', n_items)
    year_matrix, year_dim = create_item_feature_matrix(year_df, 'item_idx', 'year', n_items)
    writer_matrix, writer_dim = create_item_feature_matrix(writer_df, 'item_idx', 'writer', n_items)
    
    item_side_specs = {
        'genre': genre_matrix,
        'director': director_matrix,
        'year': year_matrix,
        'writer': writer_matrix
    }
    side_info_dims = {
        'genre': genre_dim,
        'director': director_dim,
        'year': year_dim,
        'writer': writer_dim
    }

    train_rows, train_cols = [], []
    valid_rows, valid_cols = [], []
    
    # 유저별로 그룹화하여 데이터 분리
    grouped = train_df.groupby('user_idx')
    for u_idx, group in grouped:
        items = group['item_idx'].values
        if len(items) > 5:
            train_items = items[:-1] 
            valid_items = items[-1:]
            
            train_rows.extend([u_idx] * len(train_items))
            train_cols.extend(train_items)
            valid_rows.extend([u_idx] * len(valid_items))
            valid_cols.extend(valid_items)
        else:
            train_rows.extend([u_idx] * len(items))
            train_cols.extend(items)

    # Sparse Matrix 생성
    train_matrix = sparse.csr_matrix((np.ones_like(train_rows), (train_rows, train_cols)), shape=(n_users, n_items))
    valid_matrix = sparse.csr_matrix((np.ones_like(valid_rows), (valid_rows, valid_cols)), shape=(n_users, n_items))

    user_side_matrices = {}
    for key, item_feat_matrix in item_side_specs.items():
        # Scipy Sparse Matrix와 Torch Tensor의 행렬곱
        user_feat = train_matrix @ item_feat_matrix.numpy() 
        # Tensor로 변환 및 L2 정규화
        user_feat_tensor = torch.FloatTensor(user_feat)
        user_side_matrices[key] = torch.nn.functional.normalize(user_feat_tensor, p=2, dim=1)

    # Loader 생성
    train_dataset = MovieDataset(train_matrix, user_side_matrices)
    valid_dataset = MovieDataset(train_matrix, user_side_matrices)

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    return train_loader, valid_loader, train_matrix, valid_matrix, n_items, side_info_dims, user2idx, item2idx