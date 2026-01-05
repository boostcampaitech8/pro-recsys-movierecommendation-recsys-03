import numpy as np
from tqdm import tqdm


def recall_at_k_als(model, URM, valid_df, k=10):
    recalls = []
    n_users = URM.shape[0]

    for row in tqdm(valid_df.itertuples(), total=len(valid_df)):
        u = row.user_idx
        pos_item = row.item_idx

        if u >= n_users:
            continue

        user_items = URM[u]
        if user_items.nnz == 0:
            continue

        rec_items, _ = model.recommend(
            userid=u,
            user_items=user_items,
            N=k,
            filter_already_liked_items=True
        )

        recalls.append(int(pos_item in rec_items))

    return float(np.mean(recalls))


def ndcg_at_k_als(model, URM, valid_df, k=10):
    ndcgs = []
    n_users = URM.shape[0]

    for row in tqdm(valid_df.itertuples(), total=len(valid_df)):
        u = row.user_idx
        pos_item = row.item_idx

        if u >= n_users:
            continue

        user_items = URM[u]
        if user_items.nnz == 0:
            continue

        rec_items, _ = model.recommend(
            userid=u,
            user_items=user_items,
            N=k,
            filter_already_liked_items=True
        )

        if pos_item not in rec_items:
            ndcgs.append(0.0)
        else:
            rank = np.where(rec_items == pos_item)[0][0] + 1
            ndcgs.append(1.0 / np.log2(rank + 1))

    return float(np.mean(ndcgs))
