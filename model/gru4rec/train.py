from torch.utils.data import DataLoader
import torch.optim as optim
from model import GRU4Rec
from dataset import GRU4RecDataset, collate_fn
from dataloader import SessionParallelLoader
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from utils import recall_at_k, sample_negatives, top1_loss, bpr_loss, ndcg_at_k
import numpy as np
import wandb

def train_model(model, sequences, val_items, num_items, item2idx, idx2item, n_epochs=10, batch_size=64, learning_rate=0.001):
    wandb.init(
        project="gru4rec",
        name="gru4rec_top1_bs64",
        config={
            "model": "GRU4Rec",
            "loss": "TOP1",
            "batch_size": batch_size,
            "lr": learning_rate,
            "hidden_size": model.gru.hidden_size,
            "epochs": n_epochs,
        }
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
    loader = SessionParallelLoader(sequences, batch_size)

    for epoch in range(n_epochs):
        model.train()
        h = None
        total_loss = 0
        steps = 0

        for x, y, reset in tqdm(loader, total=len(loader) , desc=f"Epoch {epoch+1}/{n_epochs}"):
            x = torch.LongTensor(x).to(device)
            y = torch.LongTensor(y).to(device)

            device = x.device
            
            if y.numel() == 0:
                print("Empty target batch, skipping step")
                continue

            if h is None:
                h = torch.zeros(1, x.size(0), model.gru.hidden_size).to(device)
            else:
                # 1) batch size가 바뀌면 hidden 재생성
                if h.size(1) != x.size(0):
                    new_h = torch.zeros(1, x.size(0), model.gru.hidden_size).to(device)
                    min_B = min(h.size(1), x.size(0))
                    new_h[:, :min_B, :] = h[:, :min_B, :]
                    h = new_h
                else:
                    h = h.detach()

                # 2) reset mask 위치 hidden 초기화
                if reset.any():
                    batch_idx = torch.nonzero(torch.tensor(reset, dtype=torch.bool, device=device), as_tuple=True)[0]
                    h[:, batch_idx, :] = 0
            logits, h = model(x, h.detach())
            optimizer.zero_grad()
            
            pos_scores = logits[torch.arange(y.size(0)), y]

            neg_items = sample_negatives(
                # batch_size=y.size(0),
                # num_items=num_items,
                # positives=y,
                # device=device
                y
            )

            neg_scores = logits[torch.arange(y.size(0)), neg_items]
            #print(pos_scores, neg_scores)
            loss = top1_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            wandb.log({
                "train/loss": loss.item(),
            })
            total_loss += loss.item()
            steps += 1
        model.eval()
        recall,ndcg = recall_at_k(model, sequences, val_items, item2idx, idx2item, k=10, device=device)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/steps:.4f}")
        print()
        wandb.log({
            "epoch": epoch + 1,
            "train/avg_loss": total_loss / steps,
            "val/recall@10": recall,
            "val/ndcg@10": ndcg
        })
    wandb.finish()


@torch.no_grad()
def create_submission(
    model,
    user_sequences,   
    item2idx,
    idx2item,
    user2idx,
    idx2user,
    top_k=10
):
    model.eval()
    device = next(model.parameters()).device
    submission = []

    for user_idx, seq in enumerate(user_sequences):

        # 빈 시퀀스 skip
        if len(seq) == 0:
            submission.append({
                "user": idx2user[user_idx],
                "item": ""
            })
            continue

        # hidden 초기화
        h = torch.zeros(
            1, 1, model.gru.hidden_size, device=device
        )

        # 🔑 전체 시퀀스를 한 step씩 GRU에 통과
        for item in seq:
            x = torch.tensor([item], device=device)  # (1,)
            emb = model.embedding(x).unsqueeze(0)    # (1, 1, embed)
            out, h = model.gru(emb, h)

        # 마지막 hidden으로 score 계산
        logits = model.fc(out.squeeze(0))  # (1, num_items)
        scores = logits.squeeze(0).cpu().numpy()

        # 이미 본 아이템 제거
        seen_items = seq
        for s in seen_items:
            scores[s] = -np.inf

        # top-k 추출
        top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        rec_items = [idx2item[i] for i in top_k_idx]

        submission.append({
            "user": idx2user[user_idx],
            "item": " ".join(map(str, rec_items))
        })

    pd.DataFrame(submission).to_csv("submission.csv", index=False)