import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


class InteractionData:
    def __init__(self, base_data_dir):
        self.train_dir = os.path.join(base_data_dir, "train")
        self.eval_dir = os.path.join(base_data_dir, "eval")

        self._load_train()
        self._load_eval()

    def _load_train(self):
        ratings = pd.read_csv(
            os.path.join(self.train_dir, "train_ratings.csv")
        )

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        ratings["user_idx"] = self.user_encoder.fit_transform(ratings["user"])
        ratings["item_idx"] = self.item_encoder.fit_transform(ratings["item"])

        self.ratings = ratings
        self.n_users = ratings["user_idx"].nunique()
        self.n_items = ratings["item_idx"].nunique()

    def _load_eval(self):
        self.sample_submission = pd.read_csv(
            os.path.join(self.eval_dir, "sample_submission.csv")
        )

    def build_urm(self):
        return csr_matrix(
            (
                np.ones(len(self.ratings)),
                (self.ratings["user_idx"], self.ratings["item_idx"])
            ),
            shape=(self.n_users, self.n_items)
        )