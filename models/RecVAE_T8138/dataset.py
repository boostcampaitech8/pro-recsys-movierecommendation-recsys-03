import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

class MovieDataset(Dataset):
    def __init__(self, user_item_matrix):
        self.matrix = user_item_matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        user_vector = self.matrix[idx].toarray().squeeze()
        return torch.FloatTensor(user_vector)

def make_matrix(df, num_users, num_items):
    row = df['user'].values
    col = df['item'].values
    data = np.ones(len(df))
    
    matrix = sparse.csr_matrix((data, (row, col)), shape=(num_users, num_items))
    return matrix