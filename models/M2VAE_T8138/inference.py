import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from datasets import get_loaders
from model import M2VAE

def main():
    # Config
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")

    # 데이터 로더 및 정보 로드
    train_loader, _, train_matrix, _, n_items, side_info_dims, user2idx, item2idx = get_loaders(config)
    
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    # 학습 시 사용한 것과 동일한 가중치 설정
    custom_weights = {
        'interaction': 1.0,
        'director': 0.9,
        'genre': 0.8,
        'year': 0.5,
        'writer': 0.5
    }

    # 모델 선언 및 가중치 로드
    model = M2VAE(
        input_dim=n_items, 
        side_info_dims=side_info_dims,
        weights=custom_weights,
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    print("Model loaded from best_model.pth")

    # 추론
    user_side_matrices = train_loader.dataset.user_side_matrices
    
    users = []
    items = []
    
    n_users = train_matrix.shape[0]
    
    with torch.no_grad():
        for u_idx in tqdm(range(n_users), desc="Inferring"):
            x = torch.FloatTensor(train_matrix[u_idx].toarray()).to(device)
            
            side_info_dict = {key: mat[u_idx].unsqueeze(0).to(device) 
                             for key, mat in user_side_matrices.items()}
            
            recon_batch, _, _ = model(x, side_info_dict)
            recon_batch = recon_batch.cpu().numpy().flatten()
            
            seen_items = train_matrix[u_idx].indices
            recon_batch[seen_items] = -np.inf
            
            top_10_idx = np.argsort(recon_batch)[-10:][::-1]
            
            users.extend([idx2user[u_idx]] * 10)
            items.extend([idx2item[i] for i in top_10_idx])

    submission = pd.DataFrame({
        'user': users, 
        'item': items
    })
    submission = submission.sort_values(by='user')
    submission.to_csv('submission.csv', index=False)
    print("Inference Complete! submission.csv saved.")

if __name__ == "__main__":
    main()