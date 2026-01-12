import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_popularity_confidence(df, num_items, power=0.75):
    item_counts = df.groupby('item_idx').size().reindex(np.arange(1, num_items + 1), fill_value=0).values
    
    confidence = np.power(item_counts, power)
    
    return confidence

class SASRecDataset(Dataset):
    def __init__(self, user_seq, max_len=200, item_num=0):
        self.user_seq = user_seq
        self.max_len = max_len
        self.item_num = item_num

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        seq = self.user_seq[index]
        
        seq_len = len(seq)
        if seq_len > self.max_len:
            input_seq = seq[-self.max_len:]
        else:
            input_seq = seq
            
        pad_len = self.max_len - len(input_seq)
        pad_seq = [0] * pad_len
        
        tokens = pad_seq + input_seq
        
        return torch.LongTensor(tokens)

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    
    df = pd.read_csv(data_path)
    
    user_ids = sorted(df['user'].unique())
    item_ids = sorted(df['item'].unique())
    
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j+1 for j, i in enumerate(item_ids)} 
    
    id2user = {i: u for u, i in user2idx.items()}
    id2item = {j: i for i, j in item2idx.items()}
    
    df['user_idx'] = df['user'].map(user2idx)
    df['item_idx'] = df['item'].map(item2idx)
    
    df = df.sort_values(by=['user_idx', 'time'])
    user_grouped = df.groupby('user_idx')['item_idx'].apply(list).reset_index()
    
    count_of_users = len(user_ids)
    user_seqs = [[] for _ in range(count_of_users)]
    for row in user_grouped.itertuples():
        user_seqs[row.user_idx] = row.item_idx
    
    pop_confidence = get_popularity_confidence(df, len(item_ids))
    
    return user_seqs, len(item_ids), user2idx, item2idx, id2user, id2item, pop_confidence

def split_data(user_seqs):
    train_seqs = []
    val_seqs = []
    
    for seq in user_seqs:
        if len(seq) < 2:
            train_seqs.append(seq)
            val_seqs.append([])
            continue
            
        val_seqs.append(seq)
        train_seqs.append(seq[:-1])
        
    return train_seqs, val_seqs

def get_data_mapping(data_path):
    df = pd.read_csv(data_path)
    df = df.sort_values(['user', 'time'])

    user_list = df['user'].unique()
    item_list = df['item'].unique()

    user2idx = {user: i for i, user in enumerate(user_list)}
    item2idx = {item: i for i, item in enumerate(item_list)}
    
    idx2item = {i: item for item, i in item2idx.items()}
    
    return user2idx, item2idx, idx2item
