import numpy as np


def auc_score(user_idx, pos_item, URM, W, n_neg=100):
    user_profile = URM[user_idx]
    if user_profile.nnz == 0:
        return None

    scores = (user_profile @ W).toarray().ravel()
    pos_score = scores[pos_item]

    seen = set(user_profile.indices)
    unseen = list(set(range(W.shape[0])) - seen)
    if not unseen:
        return None

    neg_items = np.random.choice(unseen, min(n_neg, len(unseen)), replace=False)
    return np.mean(pos_score > scores[neg_items])


def recall_at_k(user_idx, pos_item, URM, W, k):
    user_profile = URM[user_idx]
    if user_profile.nnz == 0:
        return 0

    scores = (user_profile @ W).toarray().ravel()
    scores[user_profile.indices] = -np.inf
    topk = np.argsort(scores)[-k:]
    return int(pos_item in topk)


def ndcg_at_k(user_idx, pos_item, URM, W, k):
    user_profile = URM[user_idx]
    if user_profile.nnz == 0:
        return 0.0

    scores = (user_profile @ W).toarray().ravel()
    scores[user_profile.indices] = -np.inf
    topk = np.argsort(scores)[-k:][::-1]

    if pos_item not in topk:
        return 0.0

    rank = np.where(topk == pos_item)[0][0] + 1
    return 1.0 / np.log2(rank + 1)