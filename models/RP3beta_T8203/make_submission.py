import pandas as pd
from tqdm import tqdm


def make_submission(model, URM, data, topk, save_path):
    rows = []

    for user in tqdm(data.sample_submission["user"].unique()):
        user_idx = data.user_encoder.transform([user])[0]
        recs = model.recommend(user_idx, URM, topk)
        items = data.item_encoder.inverse_transform(recs)

        for item in items:
            rows.append({"user": user, "item": item})

    pd.DataFrame(rows).to_csv(save_path, index=False)