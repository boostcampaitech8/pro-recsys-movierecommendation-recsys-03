import pandas as pd
import numpy as np
import os
import pickle

def preprocess_data(config):
    # 데이터 로드
    df = pd.read_csv(config.data.base_url + 'train_ratings.csv')
    df = df.sort_values(['user', 'time'])

    # User/Item Indexing
    user_list = df['user'].unique()
    item_list = df['item'].unique()

    user2idx = {user: i for i, user in enumerate(user_list)}
    item2idx = {item: i for i, item in enumerate(item_list)}
    
    # 역매핑
    idx2item = {i: item for item, i in item2idx.items()}

    df['user'] = df['user'].map(user2idx)
    df['item'] = df['item'].map(item2idx)

    # 매핑 정보 저장
    os.makedirs(config.data.output_dir, exist_ok=True)
    with open(os.path.join(config.data.output_dir, 'mapping.pkl'), 'wb') as f:
        pickle.dump({'user2idx': user2idx, 'item2idx': item2idx, 'idx2item': idx2item}, f)

    # 데이터 분리
    weighted_list = []
    val_list = []
    
    np.random.seed(config.seed)
    
    # 유저별로 아이템 분할
    for user, group in df.groupby('user'):
        items = group['item'].values

        np.random.shuffle(items)    # 랜덤한 1개 선택
        val_items = items[:1] 
        train_items = items[1:]     # 나머지는 전부 학습용
            
        for item in val_items:
            val_list.append([user, item])
        for item in train_items:
            weighted_list.append([user, item])

    train_df = pd.DataFrame(weighted_list, columns=['user', 'item'])
    val_df = pd.DataFrame(val_list, columns=['user', 'item'])
    
    num_users = len(user_list)
    num_items = len(item_list)

    print(f"Total Users: {num_users}, Total Items: {num_items}")
    print(f"Train/Val interactions: {len(train_df)} / {len(val_df)}")

    return train_df, val_df, num_users, num_items