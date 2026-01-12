import torch.optim as optim
import torch
import torch.nn.functional as F
from utils import recall_at_k
import numpy as np
from tqdm import tqdm
import pandas as pd

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer_enc = optim.Adam(self.model.encoder.parameters(), lr=config.train.lr)
        self.optimizer_dec = optim.Adam(self.model.decoder.parameters(), lr=config.train.lr)
        
    def loss_function(self, recon_x, x, mu, logvar, beta):
        log_softmax_var = F.log_softmax(recon_x, dim=1)
        ce_loss = -(log_softmax_var * x).sum(dim=1).mean()
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return ce_loss + beta * kl_loss

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        beta = min(self.config.train.beta, 
                   self.config.train.beta * (epoch / (self.config.train.epochs // 2)))
        gamma = getattr(self.config.train, 'gamma', 1.0)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=True)
        for batch in pbar:
            batch = batch.to(self.device)
            # Encoder Update
            for _ in range(self.config.train.enc_epochs):
                self.optimizer_enc.zero_grad()
                recon_x, mu, logvar = self.model(batch)
                loss = self.loss_function(recon_x, batch, mu, logvar, beta) * gamma
                loss.backward()
                self.optimizer_enc.step()
            # Decoder Update
            self.optimizer_dec.zero_grad()
            recon_x, mu, logvar = self.model(batch)
            loss = self.loss_function(recon_x, batch, mu, logvar, beta)
            loss.backward()
            self.optimizer_dec.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", beta=f"{beta:.3f}")
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, train_matrix, val_matrix, k=10):
        self.model.eval()
        num_users = train_matrix.shape[0]
        batch_size = self.config.train.batch_size
        recalls = []
        indices = range(0, num_users, batch_size)
        for i in tqdm(indices, desc="Evaluating", leave=False):
            end_idx = min(i + batch_size, num_users)
            batch_input = torch.FloatTensor(train_matrix[i:end_idx].toarray()).to(self.device)
            
            recon_batch, _, _ = self.model(batch_input)
            recon_batch[batch_input > 0] = -1e9
            _, top_indices = torch.topk(recon_batch, k=k, dim=1)
            top_indices = top_indices.cpu().numpy()
            for j in range(len(top_indices)):
                user_idx = i + j
                actual = val_matrix[user_idx].indices 
                if len(actual) > 0:
                    recalls.append(recall_at_k(actual, top_indices[j], k=k))
        return np.mean(recalls) if recalls else 0