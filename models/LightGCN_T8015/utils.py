import os
import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
from config import config
from torch.utils.data import Dataset

class RecDataset(Dataset):
    def __init__(self, df, num_users, num_items, confidence, train=True):
        self.df = df
        self.num_users = num_users
        self.num_items = num_items
        self.train = train
        
        self.users = torch.LongTensor(df['user'].values)
        self.items = torch.LongTensor(df['item'].values)
        
        self.user_item_set = set(zip(df['user'], df['item']))
        self.confidence = torch.tensor(confidence)
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        
        if self.train:
            neg_item = np.random.randint(0, self.num_items)
            while (user.item(), neg_item) in self.user_item_set:
                neg_item = np.random.randint(0, self.num_items)
            
            return user, pos_item, torch.tensor(neg_item), self.confidence[pos_item]
        else:
            return user, pos_item

def load_data(path, val_ratio=0.2):
    df = pd.read_csv(path)
    
    data_dir = os.path.dirname(path)
    genres_df = pd.read_csv(os.path.join(data_dir, 'genres.tsv'), sep='\t')
    directors_df = pd.read_csv(os.path.join(data_dir, 'directors.tsv'), sep='\t')
    writers_df = pd.read_csv(os.path.join(data_dir, 'writers.tsv'), sep='\t')
    
    user_unique = df['user'].unique()
    item_unique = df['item'].unique()
    
    num_users = len(user_unique)
    num_items = len(item_unique)
    
    user2id = {id: i for i, id in enumerate(user_unique)}
    item2id = {id: i for i, id in enumerate(item_unique)}
    id2user = {i: id for id, i in user2id.items()}
    id2item = {i: id for id, i in item2id.items()}
    
    df['user'] = df['user'].map(user2id)
    df['item'] = df['item'].map(item2id)
    
    valid_items = set(item_unique)
    
    genres_df = genres_df[genres_df['item'].isin(valid_items)].copy()
    directors_df = directors_df[directors_df['item'].isin(valid_items)].copy()
    writers_df = writers_df[writers_df['item'].isin(valid_items)].copy()
    
    genres_df['item'] = genres_df['item'].map(item2id)
    directors_df['item'] = directors_df['item'].map(item2id)
    writers_df['item'] = writers_df['item'].map(item2id)

    genre_unique = genres_df['genre'].unique()
    director_unique = directors_df['director'].unique()
    writer_unique = writers_df['writer'].unique()
    
    num_genres = len(genre_unique)
    num_directors = len(director_unique)
    num_writers = len(writer_unique)
    
    genre2id = {id: i for i, id in enumerate(genre_unique)}
    director2id = {id: i for i, id in enumerate(director_unique)}
    writer2id = {id: i for i, id in enumerate(writer_unique)}
    
    genres_df['genre'] = genres_df['genre'].map(genre2id)
    directors_df['director'] = directors_df['director'].map(director2id)
    writers_df['writer'] = writers_df['writer'].map(writer2id)
    
    side_info = {
        'genres': genres_df,
        'directors': directors_df,
        'writers': writers_df,
        'num_genres': num_genres,
        'num_directors': num_directors,
        'num_writers': num_writers
    }
    
    df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_ratio))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    return train_df, val_df, num_users, num_items, user2id, item2id, id2user, id2item, side_info

def create_adjacency_matrix(df, num_users, num_items, side_info):
    num_genres = side_info['num_genres']
    num_directors = side_info['num_directors']
    num_writers = side_info['num_writers']
    
    num_total_nodes = num_users + num_items + num_genres + num_directors + num_writers
    
    user_np = df['user'].values
    item_np = df['item'].values
    item_np_shifted = item_np + num_users
    
    genre_item_np = side_info['genres']['item'].values + num_users
    genre_val_np = side_info['genres']['genre'].values + num_users + num_items
    
    director_item_np = side_info['directors']['item'].values + num_users
    director_val_np = side_info['directors']['director'].values + num_users + num_items + num_genres
    
    writer_item_np = side_info['writers']['item'].values + num_users
    writer_val_np = side_info['writers']['writer'].values + num_users + num_items + num_genres + num_directors
    
    src_nodes = np.concatenate([user_np, genre_item_np, director_item_np, writer_item_np])
    dst_nodes = np.concatenate([item_np_shifted, genre_val_np, director_val_np, writer_val_np])
    
    all_src = np.concatenate([src_nodes, dst_nodes])
    all_dst = np.concatenate([dst_nodes, src_nodes])
    values = np.ones(len(all_src), dtype=np.float32)
    
    A = sp.csr_matrix((values, (all_src, all_dst)), shape=(num_total_nodes, num_total_nodes))
    
    degrees = np.array(A.sum(axis=1)).flatten()
    
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    L = D_inv_sqrt.dot(A).dot(D_inv_sqrt)
    
    coo = L.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape), num_total_nodes

def recall_at_k(actual, predicted, k=10):
    if len(actual) == 0:
        return 0.0
    
    act_set = set(actual)
    pred_set = set(predicted[:k])
    
    intersection = len(act_set & pred_set)
    return intersection / len(act_set)

def ndcg_at_k(actual, predicted, k=10):
    if len(actual) == 0:
        return 0.0
        
    act_set = set(actual)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(act_set), k))])
    dcg = 0.0
    
    for i, item in enumerate(predicted[:k]):
        if item in act_set:
            dcg += 1.0 / np.log2(i + 2)
            
    return dcg / idcg if idcg > 0 else 0.0

def get_popularity_confidence(df, num_items, alpha=20, e_val=0.3):
    item_counts = df.groupby('item').size().reindex(np.arange(num_items), fill_value=0).values
    confidence = 1 + np.log1p(alpha * (item_counts / e_val))
    return confidence