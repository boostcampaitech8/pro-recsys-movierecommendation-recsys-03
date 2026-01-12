import torch
import pickle
import os
from scipy import sparse
from models import MultiVAE
from trainers import VAETrainer
from utils import split_data
from inference import run_inference_for_catboost

def main():
    # 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists('./output'): os.makedirs('./output')
    
    # 데이터 로드 
    full_matrix = sparse.load_npz('../data/train/train_matrix.npz')
    with open('../data/train/mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    
    train_mat, val_mat = split_data(full_matrix)
    
    # 모델 및 트레이너 설정
    model = MultiVAE(input_dim=train_mat.shape[1], hidden_dim=512, latent_dim=256).to(device)
    trainer = VAETrainer(model, train_mat, val_mat, device)
    
    # 학습 루프
    best_recall = 0
    patience, counter = 30, 0
    
    for epoch in range(300):
        loss = trainer.train_epoch(epoch, batch_size=128)
        r10, r100 = trainer.evaluate(batch_size=128)
        
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | R@10: {r10:.4f} | R@100: {r100:.4f}")
        
        if r100 > best_recall:
            best_recall = r100
            torch.save(model.state_dict(), './output/multivae_best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience: break

    # 추론
    model.load_state_dict(torch.load('./output/multivae_best_model.pt'))
    run_inference_for_catboost(model, full_matrix, mapping, device, "vae_candidates_nomask", mask=False)
    run_inference_for_catboost(model, full_matrix, mapping, device, "vae_candidates_test", mask=True)

if __name__ == "__main__":
    main()