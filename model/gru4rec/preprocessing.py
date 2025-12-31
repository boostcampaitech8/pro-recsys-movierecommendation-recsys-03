import pandas as pd
import numpy as np
from collections import defaultdict
def preprocess(df):
    # df columns: user, item, time
    df = df.sort_values(["user", "time"])

    user_sequences = defaultdict(list)
    for u, i in zip(df.user, df.item):
        user_sequences[u].append(i)

    # item index (1부터, 0은 padding)
    all_items = df.item.unique()
    item2idx = {item: idx + 1 for idx, item in enumerate(all_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}

    # user index (0부터)
    all_users = list(user_sequences.keys())
    user2idx = {user: idx for idx, user in enumerate(all_users)}
    idx2user = {idx: user for user, idx in user2idx.items()}

    sequences = []
    sequence_users = []  # sequences[i]가 어떤 user인지 기록

    for user, seq in user_sequences.items():
        if len(seq) < 2:
            continue

        seq_idx = [item2idx[i] for i in seq]
        sequences.append(seq_idx)
        sequence_users.append(user2idx[user])

    num_items = len(item2idx) + 1  # padding = 0

    print("Number of users:", len(sequences))
    print("Number of items:", len(item2idx))

    return (
        sequences,
        sequence_users,
        num_items,
        item2idx,
        idx2item,
        user2idx,
        idx2user,
    )