import torch
import numpy as np
import pandas as pd
from utils import save_submission, save_candidates

# Multi-VAE 단일 모델용 (제출 파일 생성용: Top-10)
def run_inference(model, train_matrix, mapping, device, batch_size=512, top_k=10):
    model.eval()
    num_users = train_matrix.shape[0]
    id2user = {v: k for k, v in mapping['user2id'].items()}
    id2item = mapping['id2item']
    
    print("Inference for Single Multi-VAE Submission...")
    pred_list = []
    
    with torch.no_grad():
        for i in range(0, num_users, batch_size):
            end_idx = min(i + batch_size, num_users)
            batch_x = torch.FloatTensor(train_matrix[i:end_idx].toarray()).to(device)
            
            recon_batch, _, _ = model(batch_x)
            recon_batch = recon_batch.cpu().numpy()
            
            # 이미 본 아이템 제외 (Masking)
            recon_batch[train_matrix[i:end_idx].toarray() > 0] = -1e9
            
            # 상위 K개 인덱스 추출 및 정렬
            ind = np.argpartition(recon_batch, -top_k, axis=1)[:, -top_k:]
            arr_ind = recon_batch[np.arange(len(recon_batch))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind, axis=1)[:, ::-1]
            batch_pred_list = ind[np.arange(len(recon_batch))[:, None], arr_ind_argsort]
            
            pred_list.append(batch_pred_list)
            
    pred_list = np.concatenate(pred_list, axis=0)
    
    save_submission(pred_list, id2user, id2item)

def run_inference_for_catboost(model, train_matrix, mapping, device, filename, batch_size=512, top_k=100, mask=True):
    model.eval()
    num_users = train_matrix.shape[0]
    id2user = {v: k for k, v in mapping['user2id'].items()}
    id2item = mapping['id2item']
    
    results = []
    with torch.no_grad():
        for i in range(0, num_users, batch_size):
            end_idx = min(i + batch_size, num_users)
            batch_x = torch.FloatTensor(train_matrix[i:end_idx].toarray()).to(device)
            recon_batch, _, _ = model(batch_x)
            recon_batch = recon_batch.cpu().numpy()
            
            if mask:
                recon_batch[train_matrix[i:end_idx].toarray() > 0] = -1e9
            
            ind = np.argpartition(recon_batch, -top_k, axis=1)[:, -top_k:]
            rows = np.arange(len(recon_batch))[:, None]
            scores = recon_batch[rows, ind]
            
            for b_idx in range(len(recon_batch)):
                u_id = id2user[i + b_idx]
                for k_idx in range(top_k):
                    score = scores[b_idx, k_idx]
                    if score > -1e8:
                        results.append([u_id, id2item[ind[b_idx, k_idx]], score])
                        
    save_path = f'./output/{filename}.csv'
    save_candidates(results, save_path)