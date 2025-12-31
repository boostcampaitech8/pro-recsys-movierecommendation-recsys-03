import numpy as np
from scipy.sparse import csr_matrix


class RP3beta:
    def __init__(self, alpha=0.6, beta=0.4, topk=100):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk

    def fit(self, URM):
        URM = URM.tocsr()
        IRM = URM.T.tocsr()

        user_degree = np.array(URM.sum(axis=1)).ravel()
        item_degree = np.array(URM.sum(axis=0)).ravel()

        user_degree[user_degree == 0] = 1
        item_degree[item_degree == 0] = 1

        Pui = URM.multiply(1 / user_degree[:, None])
        Piu = IRM.multiply(1 / item_degree[:, None])

        W = Piu @ Pui
        W.data = np.power(W.data, self.alpha)
        W = W.multiply(np.power(item_degree, -self.beta))

        W.setdiag(0)
        W.eliminate_zeros()

        self.W = self._apply_topk(W.tocsr())

    def _apply_topk(self, W):
        data, rows, cols = [], [], []

        for i in range(W.shape[0]):
            start, end = W.indptr[i], W.indptr[i + 1]
            row_data = W.data[start:end]
            row_idx = W.indices[start:end]

            if len(row_data) > self.topk:
                topk_idx = np.argsort(row_data)[-self.topk:]
                row_data = row_data[topk_idx]
                row_idx = row_idx[topk_idx]

            data.extend(row_data)
            rows.extend([i] * len(row_data))
            cols.extend(row_idx)

        return csr_matrix((data, (rows, cols)), shape=W.shape)

    def recommend(self, user_idx, URM, k):
        scores = (URM[user_idx] @ self.W).toarray().ravel()
        scores[URM[user_idx].indices] = -np.inf
        return np.argsort(scores)[::-1][:k]
