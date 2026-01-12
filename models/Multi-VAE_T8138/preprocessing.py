import pandas as pd
import numpy as np
from scipy import sparse
import pickle
import os

def preprocess_data(data_path, save_path):
    """
    CSV 데이터를 읽어 Sparse Matrix와 Mapping 파일을 생성하고 저장합니다.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 유저 및 아이템 고유값 추출
    user_unique = df['user'].unique()
    item_unique = df['item'].unique()

    # Mapping 딕셔너리 생성
    user2id = {v: i for i, v in enumerate(user_unique)}
    item2id = {v: i for i, v in enumerate(item_unique)}
    id2item = {i: v for v, i in item2id.items()}
    # 추론 시 user id 복원을 위해 id2user도 포함하는 것이 좋습니다.
    id2user = {i: v for v, i in user2id.items()}

    # ID로 변환
    df['user_id'] = df['user'].map(user2id)
    df['item_id'] = df['item'].map(item2id)

    num_users = len(user_unique)
    num_items = len(item_unique)
    print(f"Total Users: {num_users}, Total Items: {num_items}")

    # Sparse Matrix 생성 (CSR)
    users = df['user_id'].values
    items = df['item_id'].values
    values = np.ones(len(df))
    
    all_matrix = sparse.csr_matrix((values, (users, items)), shape=(num_users, num_items))

    # 파일 저장
    matrix_save_path = os.path.join(save_path, 'train_matrix.npz')
    mapping_save_path = os.path.join(save_path, 'mapping.pkl')
    
    sparse.save_npz(matrix_save_path, all_matrix)
    
    mapping_data = {
        'user2id': user2id, 
        'item2id': item2id, 
        'id2item': id2item,
        'id2user': id2user # 추가됨
    }
    
    with open(mapping_save_path, 'wb') as f:
        pickle.dump(mapping_data, f)
        
    print(f"Preprocessing complete! Matrix saved at {matrix_save_path}")
    return all_matrix, mapping_data

if __name__ == "__main__":
    # 스크립트로 직접 실행할 때의 경로 설정
    DATA_PATH = '../data/train/train_ratings.csv'
    SAVE_PATH = '../data/train/'
    preprocess_data(DATA_PATH, SAVE_PATH)