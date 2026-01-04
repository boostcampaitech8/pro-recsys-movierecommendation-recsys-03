import torch
import numpy as np
from tqdm import tqdm

from create_sideinfo import build_side_info
import torch
import numpy as np

def recall_at_k(
    model,
    train_sequences,
    val_items,
    item2idx,
    idx2item,
    item2genres,
    item2year_bucket,
    k=10,
    device="cuda"
):
    model.eval()

    recall_sum = 0.0
    ndcg_sum = 0.0
    n_eval_users = 0

    with torch.no_grad():
        for u in range(len(val_items)):
            if len(train_sequences[u]) == 0 or len(val_items[u]) == 0:
                continue

            # 마지막 아이템만 사용 (GRU4Rec 논문 방식)
            last_item = train_sequences[u][-1]
            if last_item not in item2idx:
                continue

            x = torch.LongTensor([item2idx[last_item]]).to(device)

            genre_idx, year_idx = build_side_info(
                x,
                item2genres,
                item2year_bucket,
                device
            )

            h0 = torch.zeros(
                1, 1, model.gru.hidden_size, device=device
            )

            logits, _ = model(x, genre_idx, year_idx, h0)
            scores = logits[0].clone()

            # 이미 본 아이템 제거
            for it in train_sequences[u]:
                if it in item2idx:
                    scores[item2idx[it]] = -1e9

            topk_idx = torch.topk(scores, k).indices.cpu().numpy()
            topk_items = [idx2item[i] for i in topk_idx]

            gt_items = set(val_items[u])

            # recall@k
            hits = len(set(topk_items) & gt_items)
            recall_sum += hits / min(k, len(gt_items))

            # ndcg@k
            dcg = 0.0
            for rank, item in enumerate(topk_items):
                if item in gt_items:
                    dcg += 1.0 / np.log2(rank + 2)

            idcg = sum(
                1.0 / np.log2(i + 2)
                for i in range(min(len(gt_items), k))
            )

            ndcg_sum += dcg / idcg if idcg > 0 else 0.0

            n_eval_users += 1

    return (
        recall_sum / n_eval_users,
        ndcg_sum / n_eval_users
    )


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