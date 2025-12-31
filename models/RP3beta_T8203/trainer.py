import numpy as np
from sklearn.model_selection import train_test_split
from evaluate import auc_score, recall_at_k, ndcg_at_k


class RP3Trainer:
    def __init__(self, model, data, recall_k, ndcg_k, n_neg):
        self.model = model
        self.data = data
        self.recall_k = recall_k
        self.ndcg_k = ndcg_k
        self.n_neg = n_neg

    def offline_validate(self):
        df = self.data.ratings.copy()

        user_cnt = df.groupby("user_idx").size()
        valid_users = user_cnt[user_cnt >= 2].index
        df = df[df["user_idx"].isin(valid_users)]
        df = df.sort_values(["user_idx", "time"])

        valid_df = df.groupby("user_idx").tail(1)
        train_df = df.drop(valid_df.index)

        URM_train = self.data.build_urm()
        self.model.fit(URM_train)

        aucs, recalls, ndcgs = [], [], []

        for _, row in valid_df.iterrows():
            u, i = row["user_idx"], row["item_idx"]

            auc = auc_score(u, i, URM_train, self.model.W, self.n_neg)
            if auc is not None:
                aucs.append(auc)

            recalls.append(recall_at_k(u, i, URM_train, self.model.W, self.recall_k))
            ndcgs.append(ndcg_at_k(u, i, URM_train, self.model.W, self.ndcg_k))

        print("[Offline Validation]")
        print(f"AUC       : {np.mean(aucs):.4f}")
        print(f"Recall@10 : {np.mean(recalls):.4f}")
        print(f"NDCG@10   : {np.mean(ndcgs):.4f}")