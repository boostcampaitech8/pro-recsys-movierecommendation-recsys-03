import yaml
import torch
from datasets import get_loaders
from model import M2VAE
from trainers import Trainer
from utils import set_seed

def main():
    # Config
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['train']['seed'])
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")

    # 데이터 및 로더 준비
    train_loader, val_loader, _, val_matrix, n_items, side_info_dims, _, _ = get_loaders(config)

    custom_weights = {
        'interaction': 3.0,
        'director': 1.2,
        'genre': 1.0,
        'year': 0.8,
        'writer': 0.8
    }
    
    # 모델 선언
    model = M2VAE(
        input_dim=n_items, 
        side_info_dims=side_info_dims,
        weights=custom_weights,
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    # 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['model']['lr']))

    # Trainer 초기화
    trainer = Trainer(model, optimizer, device, config, val_matrix)

    # 학습 루프
    print(f"Start Training... (Total items: {n_items})")
    best_recall = 0
    for epoch in range(1, config['model']['epochs'] + 1):
        loss = trainer.train_epoch(train_loader)
        recall_10 = trainer.evaluate(val_loader, k=10)
        
        print(f"Epoch [{epoch}/{config['model']['epochs']}] Loss: {loss:.4f} | Recall@10: {recall_10:.4f}")
        
        if recall_10 > best_recall:
            best_recall = recall_10
            torch.save(model.state_dict(), 'best_model.pth')
            print("--- Best Model Saved ---")

if __name__ == "__main__":
    main()