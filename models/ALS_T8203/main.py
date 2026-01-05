from .config import *
from .data import *
from .model import build_als_model
from .trainer import train_als
from .evaluate import recall_at_k_als, ndcg_at_k_als


def main():
    ratings = load_train_ratings(TRAIN_PATH)

    train_df, valid_df = split_train_valid_by_last_interaction(ratings)
    train_df, valid_df, user_enc, item_enc = encode_user_item(train_df, valid_df)

    n_users = train_df["user_idx"].max() + 1
    n_items = train_df["item_idx"].max() + 1

    URM_train = build_urm(train_df, n_users, n_items)

    model = build_als_model(
        ALS_FACTORS,
        ALS_REGULARIZATION,
        ALS_ITERATIONS,
        ALS_RANDOM_STATE
    )

    model = train_als(model, URM_train)

    recall = recall_at_k_als(model, URM_train, valid_df, TOP_K)
    ndcg = ndcg_at_k_als(model, URM_train, valid_df, TOP_K)

    print("[ALS Offline Validation]")
    print(f"Recall@{TOP_K} : {recall:.4f}")
    print(f"NDCG@{TOP_K}   : {ndcg:.4f}")


if __name__ == "__main__":
    main()