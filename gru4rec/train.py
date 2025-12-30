from torch.utils.data import DataLoader
import torch.optim as optim
from model import GRU4Rec
from dataset import GRU4RecDataset, collate_fn
from dataloader import SessionParallelLoader
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from utils import recall_at_k_gru4rec_session_parallel, sample_negatives, top1_loss
import numpy as np

def train_model(model, sequences, val_items, num_items, item2idx, idx2item, n_epochs=10, batch_size=64, learning_rate=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

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
            assert y.min() >= 0 and y.max() < num_items, "target index out of range"

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

            optimizer.zero_grad()
            logits, h = model(x, h.detach())
            pos_scores = logits[torch.arange(y.size(0)), y]

            neg_items = sample_negatives(
                batch_size=y.size(0),
                num_items=num_items,
                positives=y,
                device=device
            )

            neg_scores = logits[torch.arange(y.size(0)), neg_items]

            loss = top1_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch+1}, Loss: {total_loss/steps:.4f}")
        print(recall_at_k_gru4rec_session_parallel(model, sequences, val_items, item2idx, idx2item, k=10, batch_size=64, device=device))

@torch.no_grad()
def create_submission(model, user_sequences, item2idx, idx2item, user2idx, idx2user, top_k=10):
    model.eval()
    device = next(model.parameters()).device
    submission = []

    for user, seq in enumerate(user_sequences):
        
        if len(seq) == 0:
            print("Empty sequence for user", user)
            submission.append({"user": user, "item": ""})
            continue

        # 마지막 아이템
        last_item = seq[-1]
        # if last_item not in item2idx:
        #     print(f"Last item {last_item} not in item2idx for user", user)
        #     submission.append({"user": user, "item": ""})
        #     continue

        x = torch.tensor([last_item], device=device)  # (1,)
        h = torch.zeros(1, 1, model.gru.hidden_size, device=device)

        # 👉 핵심: seq_len=1 명시
        emb = model.embedding(x)           # (1, embed)
        emb = emb.unsqueeze(0)             # (1, 1, embed)
        #print(emb.shape, h.shape)
        out, h = model.gru(emb, h)         # OK
        logits = model.fc(out.squeeze(0))  # (1, num_items)

        scores = logits.squeeze(0).cpu().numpy()

        # seen 제거
        seen = [item2idx[i] for i in seq if i in item2idx]
        scores[seen] = -np.inf

        top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        recs = [idx2item[i] for i in top_k_idx]
        user = idx2user[user]
        for r in recs:
            r = str(r)

            submission.append({
                "user": user,
                "item": r
            })

    pd.DataFrame(submission).to_csv("submission.csv", index=False)