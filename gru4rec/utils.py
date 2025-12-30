import torch
import numpy as np
from tqdm import tqdm

def recall_at_k_gru4rec_session_parallel(model, train_sequences, val_items,
                                         item2idx, idx2item, k=10,
                                         batch_size=64, device="cuda"):
    model.eval()
    recall_sum = 0.0
    n_eval_users = 0

    n_users = len(val_items)

    with torch.no_grad():
        for i in range(0, n_users, batch_size):
            batch_users = list(range(i, min(i + batch_size, n_users)))
            max_seq_len = max(len(train_sequences[u]) for u in batch_users)

            # 배치 입력
            x_batch = np.zeros((len(batch_users), max_seq_len), dtype=np.int64)
            mask_batch = np.zeros((len(batch_users), max_seq_len), dtype=bool)

            for j, u in enumerate(batch_users):
                seq = [item2idx[it] for it in train_sequences[u] if it in item2idx]
                x_batch[j, :len(seq)] = seq
                mask_batch[j, :len(seq)] = 1

            last_items = []
            for u in batch_users:
                seq = [item2idx[it] for it in train_sequences[u] if it in item2idx]
                if len(seq) == 0:
                    last_items.append(0)  # 또는 continue / skip
                else:
                    last_items.append(seq[-1])

            x_batch = torch.LongTensor(last_items).to(device)
            batch_hidden = torch.zeros(
                1, x_batch.size(0), model.gru.hidden_size, device=device
            )

            #print(x_batch.shape, batch_hidden.shape)
            logits, _ = model(x_batch, batch_hidden)

            for j, u in enumerate(batch_users):
                seq_len = mask_batch[j].sum()
                if seq_len == 0 or len(val_items[u]) == 0:
                    continue
                scores = logits[j].cpu().numpy()

                # 이미 학습에 나온 아이템 제거
                seen_items = train_sequences[u]
                for s in seen_items:
                    if s in item2idx:
                        scores[item2idx[s]] = -np.inf

                top_k_idx = np.argpartition(scores, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
                top_k_items = [idx2item[i] for i in top_k_idx]

                gt_items = val_items[u]
                hits = len(set(top_k_items) & set(gt_items))
                recall_sum += hits / min(k, len(gt_items))
                n_eval_users += 1

    return recall_sum / n_eval_users


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

def sample_negatives(batch_size, num_items, positives, device):
    neg = torch.randint(0, num_items, (batch_size,), device=device)
    mask = neg == positives
    while mask.any():
        neg[mask] = torch.randint(0, num_items, (mask.sum(),), device=device)
        mask = neg == positives
    return neg

def top1_loss(pos_scores, neg_scores):
    loss = torch.sigmoid(neg_scores - pos_scores) + torch.sigmoid(neg_scores ** 2)
    return loss.mean()
