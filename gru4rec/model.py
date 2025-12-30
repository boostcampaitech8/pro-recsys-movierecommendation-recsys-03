import torch.nn as nn

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

