
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

# ------------------
# 1. CONFIG & SEED
# ------------------
DATA_DIR = "data/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD = 0
UNK = 1
SEED = 42
PROJECT_NAME = "GRU4RecF-GridSearch"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

# ------------------
# 2. LOAD & ENCODE DATA
# ------------------
print("Loading data...")
df = pd.read_csv(f"{DATA_DIR}/train_ratings.csv").sort_values(["user", "time"])
df_g = pd.read_csv(f"{DATA_DIR}/genres.tsv", sep="\t")
df_d = pd.read_csv(f"{DATA_DIR}/directors.tsv", sep="\t")
df_w = pd.read_csv(f"{DATA_DIR}/writers.tsv", sep="\t")

item_le = LabelEncoder()
df["item"] = item_le.fit_transform(df["item"]) + 2
n_items = df["item"].nunique() + 2

user_le = LabelEncoder()
df["user"] = user_le.fit_transform(df["user"])
n_users = len(user_le.classes_)
user2items = df.groupby("user")["item"].apply(list).to_dict()

# Multi-label Feature Mapping (Side Information)
def build_multilabel(df_side, col, item_le):
    le = LabelEncoder()
    df_side[col] = le.fit_transform(df_side[col]) + 2
    feat_map = {}
    for r in df_side.itertuples():
        try:
            i_idx = item_le.transform([r.item])[0] + 2
            feat_map.setdefault(i_idx, []).append(r.__getattribute__(col))
        except: continue
    return feat_map, len(le.classes_) + 2

item2g, n_g = build_multilabel(df_g, "genre", item_le)
item2d, n_d = build_multilabel(df_d, "director", item_le)
item2w, n_w = build_multilabel(df_w, "writer", item_le)

def create_feature_matrix(n_items, n_features, item2feat):
    max_feats = max([len(v) for v in item2feat.values()]) if item2feat else 1
    matrix = np.zeros((n_items, max_feats), dtype=np.int64)
    for i, feats in item2feat.items():
        matrix[i, :len(feats)] = feats
    return torch.from_numpy(matrix).to(DEVICE)

G_MATRIX = create_feature_matrix(n_items, n_g, item2g)
D_MATRIX = create_feature_matrix(n_items, n_d, item2d)
W_MATRIX = create_feature_matrix(n_items, n_w, item2w)

# ------------------
# 3. DATASET
# ------------------
class SeqDataset(Dataset):
    def __init__(self, user2items, max_len):
        self.samples = []
        for u, items in user2items.items():
            if len(items) < 5: continue
            train_items = items[:-1] # Leave-one-out
            for t in range(1, len(train_items)):
                self.samples.append((train_items[:t], train_items[t]))
        self.max_len = max_len
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, tgt = self.samples[idx]
        seq = [PAD]*(self.max_len - len(seq[-self.max_len:])) + seq[-self.max_len:]
        return torch.tensor(seq), torch.tensor(tgt)

class EvalDataset(Dataset):
    def __init__(self, user2items, max_len):
        self.samples = []
        for u, items in user2items.items():
            if len(items) < 5: continue
            self.samples.append((items[:-1], items[-1]))
        self.max_len = max_len
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, tgt = self.samples[idx]
        seq = [PAD]*(self.max_len - len(seq[-self.max_len:])) + seq[-self.max_len:]
        return torch.tensor(seq), torch.tensor(tgt)

# ------------------
# 4. MODEL
# ------------------
class GRU4RecF(nn.Module):
    def __init__(self, hidden, emb_item, emb_side, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_item, padding_idx=PAD)
        self.g_emb = nn.EmbeddingBag(n_g, emb_side, padding_idx=PAD, mode='mean')
        self.d_emb = nn.EmbeddingBag(n_d, emb_side, padding_idx=PAD, mode='mean')
        self.w_emb = nn.EmbeddingBag(n_w, emb_side, padding_idx=PAD, mode='mean')
        self.gru = nn.GRU(emb_item + emb_side*3, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden*2, n_items)

    def forward(self, seq):
        B, T = seq.size()
        item_e = self.item_emb(seq)
        flat_seq = seq.view(-1)
        g_e = self.g_emb(G_MATRIX[flat_seq]).view(B, T, -1)
        d_e = self.d_emb(D_MATRIX[flat_seq]).view(B, T, -1)
        w_e = self.w_emb(W_MATRIX[flat_seq]).view(B, T, -1)
        x = self.dropout(torch.cat([item_e, g_e, d_e, w_e], dim=-1))
        out, _ = self.gru(x)
        idx = (seq != PAD).sum(dim=1) - 1
        h = self.dropout(out[torch.arange(B), idx])
        return self.fc(h)

def bpr_loss(pos, neg):
    return -torch.mean(F.logsigmoid(pos - neg))

@torch.no_grad()
def recall_at_10(model, loader):
    model.eval()
    hits, total = 0, 0
    for seq, tgt in loader:
        seq, tgt = seq.to(DEVICE), tgt.to(DEVICE)
        logits = model(seq)
        logits.scatter_(1, seq, -1e9)
        logits[:, [PAD, UNK]] = -1e9
        topk = torch.topk(logits, 10, dim=1).indices
        hits += (topk == tgt.unsqueeze(1)).any(dim=1).sum().item()
        total += tgt.size(0)
    return hits / total

# ------------------
# 5. OPTUNA OBJECTIVE (Grid Search 적용)
# ------------------
def objective(trial):
    lr = 0.0007
    dropout = 0.15

    hidden = trial.suggest_categorical("hidden", [128, 256])
    emb_item = trial.suggest_categorical("emb_item", [128, 256])
    emb_side = trial.suggest_categorical("emb_side", [32, 48])
    max_len = trial.suggest_categorical("max_len", [50])

    train_ldr = DataLoader(
        SeqDataset(user2items, max_len),
        batch_size=1024,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    eval_ldr = DataLoader(EvalDataset(user2items, max_len), batch_size=2048, num_workers=6, pin_memory=True)

    model = GRU4RecF(hidden, emb_item, emb_side, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val = 0
    
    for epoch in range(1, 6):
        model.train()
        epoch_loss = 0
        for seq, pos in train_ldr:
            seq, pos = seq.to(DEVICE), pos.to(DEVICE)
            neg = torch.randint(2, n_items, pos.shape, device=DEVICE)
            logits = model(seq)
            loss = bpr_loss(logits.gather(1, pos.unsqueeze(1)), logits.gather(1, neg.unsqueeze(1)))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        
        val_score = recall_at_10(model, eval_ldr)
        
        wandb.log({
            "trial": trial.number,
            "epoch": epoch,
            "recall@10": val_score,
            "loss": epoch_loss / len(train_ldr),
            "hidden": hidden,
            "emb_item": emb_item,
            "emb_side": emb_side
        })

        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_val = max(best_val, val_score)

    return best_val

# ------------------
# 6. MAIN EXECUTION
# ------------------
if __name__ == "__main__":
    wandbc = WeightsAndBiasesCallback(metric_name="recall@10", wandb_kwargs={"project": PROJECT_NAME})

    search_space = {
        "hidden": [128, 256],
        "emb_item": [128, 256],
        "emb_side": [32, 48],
        "max_len": [50]
    }
    
    total_combinations = 1
    for v in search_space.values():
        total_combinations *= len(v)

    print(f"\n[Step 1] Optuna 그리드 서칭 시작 ({total_combinations} 번)")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        storage="sqlite:///optuna_grid_search.db",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=total_combinations, callbacks=[wandbc])

    print(f"\nBest Params: {study.best_params}")
    bp = study.best_params

    wandb.init(project=PROJECT_NAME, name="Best_Model", config=bp)

    print("\n[Step 2] best model로 다시 학습 시작")
    final_train_ldr = DataLoader(SeqDataset(user2items, bp['max_len']), batch_size=1024, shuffle=True, num_workers=6, pin_memory=True)
    final_eval_ldr = DataLoader(EvalDataset(user2items, bp['max_len']), batch_size=1024, num_workers=6, pin_memory=True)

    final_model = GRU4RecF(bp['hidden'], bp['emb_item'], bp['emb_side'], 0.15).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=bp['lr'])
    
    best_r10 = 0
    
    for epoch in range(1, 40):
        final_model.train()
        pbar = tqdm(final_train_ldr, desc=f"Final Epoch {epoch}")
        for seq, pos in pbar:
            seq, pos = seq.to(DEVICE), pos.to(DEVICE)
            neg = torch.randint(2, n_items, pos.shape, device=DEVICE)
            logits = final_model(seq)
            loss = bpr_loss(logits.gather(1, pos.unsqueeze(1)), logits.gather(1, neg.unsqueeze(1)))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            wandb.log({"final_train_loss": loss.item(), "epoch": epoch})
        
        cur_r10 = recall_at_10(final_model, final_eval_ldr)
        wandb.log({"final_recall@10": cur_r10, "epoch": epoch})
        
        if cur_r10 > best_r10:
            best_r10 = cur_r10
            torch.save(final_model.state_dict(), "best_tuned_model.pt")

    # ------------------
    # 7. LOGIT EXPORT
    # ------------------
    print("\n[Step 3] Exporting Best Logits...")
    final_model.load_state_dict(torch.load("best_tuned_model.pt"))
    
    def get_all_seqs(user2items, max_len, n_users):
        all_seqs = np.zeros((n_users, max_len), dtype=np.int64)
        for u in range(n_users):
            items = user2items.get(u, [])
            if not items: continue
            seq = items[-max_len:]
            all_seqs[u, -len(seq):] = seq
        return all_seqs

    user_all_sequences = get_all_seqs(user2items, bp['max_len'], n_users)
    all_logits = np.zeros((n_users, n_items), dtype=np.float32)
    
    final_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_users, 512), desc="Generating Logits"):
            end_idx = min(i + 512, n_users)
            batch_seqs = torch.from_numpy(user_all_sequences[i:end_idx]).to(DEVICE)
            logits = final_model(batch_seqs)
            all_logits[i:end_idx] = logits.cpu().numpy()

    np.save("user_item_logits_best.npy", all_logits)
    wandb.finish()
    print("Wandb Updated")
