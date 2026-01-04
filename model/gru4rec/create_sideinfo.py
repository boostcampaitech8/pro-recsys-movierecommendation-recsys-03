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

def build_side_info(batch_items, item2genres, item2year_bucket, device):
    """
    batch_items: (B,) torch.LongTensor (item index)
    """

    # year
    year_idx = torch.tensor(
        [item2year_bucket[int(i)] for i in batch_items],
        device=device
    )

    # genre (패딩)
    genre_lists = [item2genres[int(i)] for i in batch_items]
    max_g = max(len(g) for g in genre_lists)

    genre_idx = torch.zeros(len(batch_items), max_g, dtype=torch.long)

    for i, g in enumerate(genre_lists):
        genre_idx[i, :len(g)] = torch.tensor(g)

    genre_idx = genre_idx.to(device)

    return genre_idx, year_idx

year_df = pd.read_csv("D:\\movierec\\data\\train\\years.tsv")  # item, year

item2year_bucket = {
    int(row.item): year_to_bucket(int(row.year))
    for _, row in year_df.iterrows()
}

genre_df = pd.read_csv("D:\\movierec\\data\\train\\genres.tsv")  # item, genre

genres = sorted(genre_df["genre"].unique())
genre2idx = {g: i for i, g in enumerate(genres)}
idx2genre = {i: g for g, i in genre2idx.items()}



item2genres = defaultdict(list)

for _, row in genre_df.iterrows():
    item = int(row.item)
    genre = row.genre
    item2genres[item].append(genre2idx[genre])
