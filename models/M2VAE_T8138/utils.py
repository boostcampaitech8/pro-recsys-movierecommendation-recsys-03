import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_item_feature_matrix(df, item_col, feature_col, n_items):
    unique_features = df[feature_col].unique()
    feat2idx = {f: i for i, f in enumerate(unique_features)}
    
    # 행, 열 위치를 리스트로 추출
    row_idx = df[item_col].values
    col_idx = df[feature_col].map(feat2idx).values
    
    # Sparse하게 생성 후 Dense로 변환 (메모리 효율적)
    matrix = np.zeros((n_items, len(unique_features)))
    matrix[row_idx, col_idx] = 1
    
    return torch.FloatTensor(matrix), len(unique_features)

def recall_at_k(recon_batch, ground_truth, k=20):
    _, next_items = torch.topk(recon_batch, k)
    next_items = next_items.cpu().detach().numpy()
    
    hits = 0
    for i in range(len(next_items)):
        true_set = set(np.where(ground_truth[i] > 0)[0])
        pred_set = set(next_items[i])
        if len(true_set & pred_set) > 0:
            hits += len(true_set & pred_set) / len(true_set)
            
    return hits / len(next_items)