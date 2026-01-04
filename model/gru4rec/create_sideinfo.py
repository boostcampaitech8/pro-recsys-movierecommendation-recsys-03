import pandas as pd
from collections import defaultdict
import torch

def year_to_bucket(year):
    if year < 1930: return 0
    elif year < 1950: return 1
    elif year < 1970: return 2
    elif year < 1990: return 3
    elif year < 2000: return 4
    elif year < 2005: return 5
    else: return 6

def build_item2year_bucket(item2idx, num_items, device):
    year_df = pd.read_csv(
        "D:\\movierec\\data\\train\\years.tsv",
        sep="\t"
    ) 

    item2year_bucket = torch.zeros(num_items, dtype=torch.long)

    for _, row in year_df.iterrows():
        raw_item = int(row["item"])

        if raw_item not in item2idx:
            continue

        item_idx = item2idx[raw_item]
        item2year_bucket[item_idx] = year_to_bucket(int(row["year"]))

    return item2year_bucket.to(device)


def build_item2genres(item2idx):
    genre_df = pd.read_csv("D:\\movierec\\data\\train\\genres.tsv", sep="\t")  # item, genre

    genres = sorted(genre_df["genre"].unique())
    genre2idx = {g: i for i, g in enumerate(genres)}
    idx2genre = {i: g for g, i in genre2idx.items()}

    item2genres = defaultdict(list)

    for _, row in genre_df.iterrows():
        item = int(row["item"])
        genre = row.genre
        item2genres[item].append(genre2idx[genre])

    itemidx2genres = {}

    for raw_item, item_idx in item2idx.items():
        itemidx2genres[item_idx] = item2genres.get(raw_item, [])

    return itemidx2genres, genre2idx, idx2genre

def build_side_info(
    batch_items,          
    item2genres,
    item2year_bucket,     
    device
):
    if not torch.is_tensor(batch_items):
        batch_items = torch.tensor(batch_items, dtype=torch.long)

    batch_items = batch_items.to(device)

    year_idx = item2year_bucket[batch_items]

    batch_items_list = batch_items.tolist()
    genre_lists = [item2genres.get(i, []) for i in batch_items_list]

    max_g = max(1, max(len(g) for g in genre_lists))

    genre_idx = torch.zeros(
        len(batch_items_list), max_g,
        dtype=torch.long,
        device=device
    )

    for i, g in enumerate(genre_lists):
        if g:
            genre_idx[i, :len(g)] = torch.tensor(g, device=device)

    return genre_idx, year_idx
