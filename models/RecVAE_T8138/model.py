import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class RecVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=600, latent_dim=200, dropout=0.5):
        super(RecVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, latent_dim * 2) # mean and logvar
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # L2 Normalize input
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        
        recon_x = self.decoder(z)
        return recon_x, mu, logvar