import torch
import torch.nn.functional as F
import numpy as np
from utils import recall_at_k

class Trainer:
    def __init__(self, model, optimizer, device, config, val_matrix=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.val_matrix = val_matrix
        
        # 하이퍼파라미터
        self.total_anneal_steps = config['train'].get('total_anneal_steps', 20000)
        self.max_beta = config['train'].get('beta', 0.2)
        self.update_count = 0

    def loss_function(self, recon_x, x, mu, logvar, beta):
        # Multinomial Likelihood: 유저가 본 아이템의 확률을 극대화
        log_softmax_var = F.log_softmax(recon_x, dim=1)
        neg_ll = -torch.mean(torch.sum(log_softmax_var * x, dim=1))

        # KL Divergence: Latent 공간을 정규분포에 가깝게 규제
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        return neg_ll + beta * kld

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for x, side_dict in train_loader:
            x = x.to(self.device)
            side_dict = {k: v.to(self.device) for k, v in side_dict.items()}
            
            # KL Annealing
            if self.total_anneal_steps > 0:
                beta = min(self.max_beta, self.max_beta * self.update_count / self.total_anneal_steps)
            else:
                beta = self.max_beta
            
            self.optimizer.zero_grad()
            recon_x, mu, logvar = self.model(x, side_dict)
            
            loss = self.loss_function(recon_x, x, mu, logvar, beta)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.update_count += 1
            
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader, k=20):
        self.model.eval()
        all_recalls = []
        
        start_idx = 0
        for x, side_dict in val_loader:
            x = x.to(self.device)
            side_dict = {k: v.to(self.device) for k, v in side_dict.items()}
            
            recon_batch, _, _ = self.model(x, side_dict)
            recon_batch[x > 0] = -np.inf
            
            batch_size = x.shape[0]
            targets = self.val_matrix[start_idx : start_idx + batch_size].toarray()
            
            recall = recall_at_k(recon_batch, targets, k)
            all_recalls.append(recall)
            start_idx += batch_size
            
        return np.mean(all_recalls)