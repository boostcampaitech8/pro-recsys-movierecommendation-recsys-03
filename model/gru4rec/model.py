import torch.nn as nn
import torch

class GRU4Rec(nn.Module):
    def __init__(self, num_items, embed_dim=100, hidden_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=False)
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, x, h):
        emb = self.embedding(x).unsqueeze(0)  # (1,B,E)
        out, h = self.gru(emb, h)
        logits = self.fc(out.squeeze(0))
        return logits, h


class GRU4RecMovieSide(nn.Module):
    def __init__(
        self,
        num_items,
        num_genres,
        num_year_buckets,
        item_dim=64,
        genre_dim=16,
        year_dim=8,
        hidden_dim=100
    ):
        super().__init__()

        self.item_emb = nn.Embedding(num_items, item_dim)
        self.genre_emb = nn.Embedding(num_genres, genre_dim)
        self.year_emb = nn.Embedding(num_year_buckets, year_dim)

        input_dim = item_dim + genre_dim + year_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=False
        )

        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, item, genre_idx, year_idx, h):
        # item: (B,)
        # genre_idx: list of lists OR padded tensor
        # year_idx: (B,)

        item_e = self.item_emb(item)   # (B, item_dim)
        year_e = self.year_emb(year_idx)  # (B, year_dim)

        # 장르 평균 pooling
        genre_e = self.genre_emb(genre_idx)      # (B, G, genre_dim)
        genre_e = genre_e.mean(dim=1)             # (B, genre_dim)

        x = torch.cat([item_e, genre_e, year_e], dim=-1)
        x = x.unsqueeze(0)  # (1, B, input_dim)

        out, h = self.gru(x, h)
        logits = self.fc(out.squeeze(0))
        return logits, h
    
    
