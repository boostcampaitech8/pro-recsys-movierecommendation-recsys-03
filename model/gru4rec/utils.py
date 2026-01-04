import torch
import numpy as np
from tqdm import tqdm

def recall_at_k(
    model, train_sequences, val_items,
    item2idx, idx2item,
    k=10, device="cuda"
):
    model.eval()
    recall_sum = 0.0
    ndcg_sum = 0.0
    n_eval = 0

    with torch.no_grad():
        for u in range(len(val_items)):
            # --- 1. train sequence 준비 ---
            seq = train_sequences[u]
            if len(seq) == 0 or len(val_items[u]) == 0:
                continue

            x = torch.LongTensor(seq).unsqueeze(1).to(device)

            h0 = torch.zeros(1, 1, model.gru.hidden_size, device=device)

            emb = model.embedding(x)          
            out, h = model.gru(emb, h0)

            logits = model.fc(out[-1])        
            scores = logits.squeeze(0).cpu().numpy()

            for s in seq:
                scores[s] = -np.inf

            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

            top_k_items = [
                idx2item[i] for i in top_k_idx if i in idx2item
            ]

            gt_items = [idx2item[i] for i in val_items[u]]

            hits = len(set(top_k_items) & set(gt_items))
            recall_sum += hits / min(k, len(gt_items))

            ndcg = ndcg_at_k(top_k_items, gt_items, k)
            ndcg_sum += ndcg
            n_eval += 1

    return recall_sum / n_eval, ndcg_sum / n_eval

def ndcg_at_k(top_k_items, gt_items, k):
    dcg = 0.0
    for i, item in enumerate(top_k_items[:k]):
        if item in gt_items:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(gt_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg

def create_val_items(sequences, k=1):
    train_sequences = []
    val_items = []

    for seq in sequences:
        if len(seq) <= k:
            train_sequences.append(seq[:])
            val_items.append([])
            continue
        train_sequences.append(seq[:-k])
        val_items.append(seq[-k:])

    return train_sequences, val_items

def sample_negatives(y):
    # neg = torch.randint(1, num_items, (batch_size,), device=device)
    # mask = neg == positives
    # while mask.any():
    #     neg[mask] = torch.randint(1, num_items, (mask.sum(),), device=device)
    #     mask = neg == positives
    # return neg
    idx = torch.randperm(y.size(0), device=y.device)
    return y[idx]

def top1_loss(pos_scores, neg_scores):
    loss = torch.sigmoid(neg_scores - pos_scores) + torch.sigmoid(neg_scores ** 2)
    return loss.mean()

def bpr_loss(pos_scores, neg_scores, eps=1e-8):
    # pos_scores, neg_scores: (batch_size,)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + eps).mean()

import numpy as np
import torch

@torch.no_grad()
def save_logits(
    model,
    user_sequences,
    save_path="logits.npy"
):
    print("Saving logits...")
    model.eval()
    device = next(model.parameters()).device

    all_logits = []

    for user, seq in enumerate(user_sequences):
        if len(seq) == 0:
            continue

        # 마지막 아이템 (이미 item index라고 가정)
        last_item = seq[-1]

        x = torch.tensor([last_item], device=device)  # (1,)
        h = torch.zeros(1, 1, model.gru.hidden_size, device=device)

        # GRU forward (seq_len = 1)
        emb = model.embedding(x).unsqueeze(0)  # (1, 1, embed)
        out, _ = model.gru(emb, h)
        logits = model.fc(out.squeeze(0))      # (1, num_items)

        all_logits.append(logits.squeeze(0).cpu().numpy())

    all_logits = np.stack(all_logits, axis=0)
    np.save(save_path, all_logits)

    print(f"[Saved] logits shape = {all_logits.shape}")