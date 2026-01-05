import os
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .model import build_als_model
from .trainer import train_als
from .config import *


def make_submission():
    train_ratings = pd.read_csv(
        os.path.join(TRAIN_PATH, "train_ratings.csv")
    )

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    train_ratings["user_idx"] = user_encoder.fit_transform(train_ratings["user"])
    train_ratings["item_idx"] = item_encoder.fit_transform(train_ratings["item"])

    n_users = train_ratings["user_idx"].nunique()
    n_items = train_ratings["item_idx"].nunique()

    URM_full = csr_matrix(
        (
            np.ones(len(train_ratings)),
            (train_ratings["user_idx"], train_ratings["item_idx"])
        ),
        shape=(n_users, n_items)
    )

    model = build_als_model(
        ALS_FACTORS,
        ALS_REGULARIZATION,
        ALS_ITERATIONS,
        ALS_RANDOM_STATE
    )
    model = train_als(model, URM_full)

    sample_submission = pd.read_csv(
        os.path.join(EVAL_PATH, "sample_submission.csv")
    )

    submission_rows = []

    for user in sample_submission["user"].unique():
        user_idx = user_encoder.transform([user])[0]

        rec_items, _ = model.recommend(
            user_idx,
            URM_full[user_idx],
            N=TOP_K,
            filter_already_liked_items=True
        )

        rec_items = item_encoder.inverse_transform(rec_items)

        submission_rows.extend(
            [{"user": user, "item": item} for item in rec_items]
        )

    submission_df = pd.DataFrame(submission_rows)
    output_path = os.path.join(OUTPUT_PATH, "ALS_submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
