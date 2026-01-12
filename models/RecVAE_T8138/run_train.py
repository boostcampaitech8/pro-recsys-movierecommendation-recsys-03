from utils import set_seed, recall_at_k
from dataset import make_matrix, MovieDataset
from model import RecVAE
from trainer import Trainer
from torch.utils.data import DataLoader
from box import ConfigBox
from preprocessing import preprocess_data
import pandas as pd
import yaml
import torch
import os
from inference import generate_submission

def main():
    with open('configs/recvae.yaml', 'r') as f:
        config = ConfigBox(yaml.safe_load(f))

    set_seed(config.seed)

    if config.train.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("="*30)
    print(f"프로젝트: Movie Recommendation")
    print(f"모델: RecVAE")
    print(f"Batch Size: {config.train.batch_size}")
    print(f"사용 디바이스: {device}")
    print(f"시드 값: {config.seed}")
    print("="*30)

    train_df, val_df, num_users, num_items = preprocess_data(config)

    train_matrix = make_matrix(train_df, num_users, num_items)
    val_matrix = make_matrix(val_df, num_users, num_items)

    train_dataset = MovieDataset(train_matrix)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.train.batch_size, 
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"데이터셋 준비 완료: {train_matrix.shape[0]}명의 유저가 훈련셋에 있습니다.")

    # 모델 초기화
    model = RecVAE(
        input_dim=num_items,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        dropout=config.model.dropout
    ).to(device)

    # 트레이너 초기화
    trainer = Trainer(model, config, device)

    # 학습 루프
    print("학습을 시작합니다...")
    best_recall = 0
    for epoch in range(1, config.train.epochs + 1):
        # 학습
        avg_loss = trainer.train_epoch(train_loader, epoch)
        
        # 검증
        if epoch % 5 == 0 or epoch == 1:
            # 평가 시에는 train_matrix(입력용)와 val_matrix(정답용)를 함께 넣음
            recall = trainer.evaluate(train_matrix, val_matrix, k=10)
            print(f"Epoch [{epoch}/{config.train.epochs}] Loss: {avg_loss:.4f} | Recall@10: {recall:.4f}")
            
            # Best 모델 저장 로직 
            if recall > best_recall:
                best_recall = recall
                torch.save(model.state_dict(), 'best_model.pth')
                print("--- Best 모델 저장 완료 ---")

    print("학습 완료! 최종 추론을 시작합니다.")
        
    # 최고 성능 모델 로드
    model.load_state_dict(torch.load('best_model.pth'))
        
    # 제출용 데이터프레임 생성
    mapping_path = os.path.join(config.data.output_dir, 'mapping.pkl')
    submission_df = generate_submission(
        model, 
        train_matrix, 
        config, 
        device, 
        mapping_path
    )

    # CSV 저장
    submission_df.to_csv(config.data.submission_path, index=False)
    print(f"제출 파일 저장 완료: {config.data.submission_path}")

if __name__ == "__main__":
    main()