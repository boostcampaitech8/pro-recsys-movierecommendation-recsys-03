import torch
import pandas as pd
from tqdm import tqdm
import os
import pickle

@torch.no_grad()
def generate_submission(model, train_matrix, config, device, mapping_path):
    model.eval()
    
    # 매핑 정보 로드
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    idx2user = {i: user for user, i in mapping['user2idx'].items()}
    idx2item = mapping['idx2item']

    num_users = train_matrix.shape[0]
    batch_size = config.train.batch_size
    results = []
    
    indices = range(0, num_users, batch_size)
    for i in tqdm(indices, desc="Final Inference"):
        end_idx = min(i + batch_size, num_users)
        batch_input = torch.FloatTensor(train_matrix[i:end_idx].toarray()).to(device)
        
        recon_batch, _, _ = model(batch_input)
        
        # Masking: 이미 본 아이템 제외
        recon_batch[batch_input > 0] = -1e9
        
        # Top-10 추출
        _, top_indices = torch.topk(recon_batch, k=10, dim=1)
        top_indices = top_indices.cpu().numpy()
        
        for j, pred_items in enumerate(top_indices):
            user_id = idx2user[i + j]
            for item_idx in pred_items:
                results.append((user_id, idx2item[item_idx]))
    
    submission_df = pd.DataFrame(results, columns=['user', 'item'])
    return submission_df