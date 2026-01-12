import pandas as pd
import numpy as np
import torch
import gc
import os
import argparse
from sklearn.preprocessing import LabelEncoder

### MANUAL of hybridEASE.py
### put text into your terminal
# python hybridEASE.py --ITEM_BEST 660 --USER_BEST 4573

parser = argparse.ArgumentParser()
parser.add_argument('--ITEM_BEST', type=int)
parser.add_argument('--USER_BEST', type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITEM_BEST_L2 = args.ITEM_BEST
USER_BEST_L2 = args.USER_BEST

def force_clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

df = pd.read_csv("train/train_ratings.csv")
df = df.drop('time', axis=1)

user_enc, item_enc = LabelEncoder(), LabelEncoder()
df['u_idx'] = user_enc.fit_transform(df['user'])
df['i_idx'] = item_enc.fit_transform(df['item'])

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)
user_ids = user_enc.classes_
item_ids = item_enc.classes_


rows = torch.from_numpy(df['u_idx'].values)
cols = torch.from_numpy(df['i_idx'].values)
X = torch.zeros((n_users, n_items), dtype=torch.float32)
X[rows, cols] = 1.0
X_gpu = X.to(device)

G_item = X_gpu.t() @ X_gpu
G_item[torch.arange(n_items), torch.arange(n_items)] += ITEM_BEST_L2
P_item = torch.linalg.inv(G_item)
B_item = P_item / (-torch.diag(P_item))
B_item.diagonal().fill_(0)

scores_item = X_gpu @ B_item

del G_item, P_item, B_item
force_clear_gpu()

G_user = X_gpu @ X_gpu.t()
G_user[torch.arange(n_users), torch.arange(n_users)] += USER_BEST_L2
P_user = torch.linalg.inv(G_user)
B_user = P_user / (-torch.diag(P_user)).view(-1, 1)
B_user.diagonal().fill_(0)

scores_user = B_user @ X_gpu

del G_user, P_user, B_user
force_clear_gpu()

# ensemble_and_save_npy
def save_scores_to_npy(final_scores, filename):
    """
    final_scores를 받아서 .npy 파일로 저장합니다.
    """
    print(f"Starting to save scores to {filename}...")

    # 1. PyTorch Tensor를 Numpy Array로 변환
    if torch.is_tensor(final_scores):
        final_scores_np = final_scores.detach().cpu().numpy()
    else:
        final_scores_np = final_scores

    # 2. .npy 파일로 저장
    if not filename.endswith('.npy'):
        filename += '.npy'

    np.save(filename, final_scores_np)
    print(f"Done: {filename} (Shape: {final_scores_np.shape})")

# itme-base, user-based, iu55-based, iu73-based, iu37based
save_scores_to_npy(scores_item.clone(), "../logits/EASE_logit_item.npy")
save_scores_to_npy(scores_user.clone(), "../logits/EASE_logit_user.npy")
save_scores_to_npy(0.5 * scores_item + 0.5 * scores_user, "../logits/EASE_logit_55.npy")
save_scores_to_npy(0.7 * scores_item + 0.3 * scores_user, "../logits/EASE_logit_73.npy")
save_scores_to_npy(0.3 * scores_item + 0.7 * scores_user, "../logits/EASE_logit_37.npy")

print("\nAll .npy created successfully!")
