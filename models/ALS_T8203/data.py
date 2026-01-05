import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


def load_train_ratings(train_path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(train_path, "train_ratings.csv"))


def split_train_valid_by_last_interaction(df: pd.DataFrame):
    user_cnt = df.groupby("user").size()
    valid_users = user_cnt[user_cnt >= 2].index

    df = df[df["user"].isin(valid_users)]
    df = df.sort_values(["user", "time"])

    valid_df = df.groupby("user").tail(1).copy()
    train_df = df.drop(valid_df.index).copy()

    return train_df, valid_df


def encode_user_item(train_df, valid_df=None):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    train_df["user_idx"] = user_encoder.fit_transform(train_df["user"])
    train_df["item_idx"] = item_encoder.fit_transform(train_df["item"])

    if valid_df is not None:
        valid_df = valid_df[
            valid_df["user"].isin(user_encoder.classes_) &
            valid_df["item"].isin(item_encoder.classes_)
        ].copy()

        valid_df["user_idx"] = user_encoder.transform(valid_df["user"])
        valid_df["item_idx"] = item_encoder.transform(valid_df["item"])

    return train_df, valid_df, user_encoder, item_encoder


def build_urm(df: pd.DataFrame, n_users: int, n_items: int):
    return csr_matrix(
        (
            np.ones(len(df), dtype=np.float32),
            (df["user_idx"].values, df["item_idx"].values)
        ),
        shape=(n_users, n_items)
    )
