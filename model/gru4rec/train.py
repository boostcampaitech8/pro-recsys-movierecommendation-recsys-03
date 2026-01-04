from torch.utils.data import DataLoader
import torch.optim as optim
from model import GRU4Rec
from dataset import GRU4RecDataset, collate_fn
from dataloader import SessionParallelLoader
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from utils import recall_at_k, sample_negatives, top1_loss, bpr_loss, ndcg_at_k, save_logits
from create_sideinfo import build_side_info, build_item2year_bucket, build_item2genres
import numpy as np
import wandb

def train_model(model, sequences, val_items, num_items, item2idx, idx2item, user2idx, idx2user,n_epochs=10, batch_size=64, learning_rate=0.001):
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
    best_recall = -1

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
    loader = SessionParallelLoader(sequences, batch_size)

    item2year_bucket = build_item2year_bucket(item2idx, num_items, device)
    item2genres, genre2idx, idx2genre = build_item2genres(item2idx)

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
            
            # side info
            genre_idx, year_idx = build_side_info(
                batch_items=x,
                item2genres=item2genres,
                item2year_bucket=item2year_bucket,
                device=device
            )

            assert genre_idx.min() >= 0
            assert genre_idx.max() < 18

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

            logits, h = model(x, genre_idx, year_idx, h.detach())
            optimizer.zero_grad()
            
            pos_scores = logits[torch.arange(y.size(0)), y]

            neg_items = sample_negatives(y)

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
        recall,ndcg = recall_at_k(model, sequences, val_items, item2idx, idx2item,item2genres, item2year_bucket, k=10, device=device)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/steps:.4f}")
        print()
        wandb.log({
            "epoch": epoch + 1,
            "train/avg_loss": total_loss / steps,
            "val/recall@10": recall,
            "val/ndcg@10": ndcg
        })
        if recall > best_recall:
            best_recall = recall
            torch.save({
                "model_state_dict": model.state_dict(),
                "item2idx": item2idx,
                "idx2item": idx2item,
                "user2idx": user2idx,
                "idx2user": idx2user,
            }, f"gru4rec_best_{batch_size}.pt")

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
    print("Creating submission...")
    model.eval()
    device = next(model.parameters()).device
    submission = []

    for user_idx, seq in enumerate(tqdm(user_sequences)):

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

        for i in rec_items:
            submission.append({
                "user": idx2user[user_idx],
                "item": i
            })
    save_logits(
        model,
        user_sequences,
        save_path="submission\\submission_logits.npy"
    )
    pd.DataFrame(submission).to_csv("submission\\submission.csv", index=False)

@torch.no_grad()
def create_submission_sideinfo(
    model,
    user_sequences,
    item2idx,
    idx2item,
    user2idx,
    idx2user,
    item2genres,
    item2year_bucket,
    top_k=10
):
    print("Creating submission...")
    model.eval()
    device = next(model.parameters()).device
    submission = []

    for user_idx, seq in enumerate(tqdm(user_sequences)):

        if len(seq) == 0:
            submission.append({
                "user": idx2user[user_idx],
                "item": ""
            })
            continue

        # raw item → index
        seq_idx = [item2idx[i] for i in seq if i in item2idx]
        if len(seq_idx) == 0:
            submission.append({
                "user": idx2user[user_idx],
                "item": ""
            })
            continue

        h = torch.zeros(1, 1, model.gru.hidden_size, device=device)

        for item_idx in seq_idx:
            x = torch.LongTensor([item_idx]).to(device)

            genre_idx, year_idx = build_side_info(
                x,
                item2genres,
                item2year_bucket,
                device
            )

            logits, h = model(x, genre_idx, year_idx, h)

        scores = logits[0].clone()

        # 이미 본 아이템 제거
        scores[seq_idx] = -1e9

        topk_idx = torch.topk(scores, top_k).indices.cpu().numpy()
        rec_items = [idx2item[i] for i in topk_idx]

        for it in rec_items:
            submission.append({
                "user": idx2user[user_idx],
                "item": it
            })

    pd.DataFrame(submission).to_csv(
        "submission/submission_sideinfo.csv",
        index=False
    )
