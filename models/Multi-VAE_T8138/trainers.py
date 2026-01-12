import torch
import torch.optim as optim
import numpy as np
from modules import loss_fn, get_recall

class VAETrainer:
    def __init__(self, model, train_matrix, val_matrix, device, lr=1e-3):
        self.model = model
        self.train_matrix = train_matrix
        self.val_matrix = val_matrix
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        self.num_users = train_matrix.shape[0]

    def train_epoch(self, epoch, batch_size):
        self.model.train()
        total_loss = 0
        idxlist = np.arange(self.num_users)
        np.random.shuffle(idxlist)
        
        for batch_idx in range(0, self.num_users, batch_size):
            end_idx = min(batch_idx + batch_size, self.num_users)
            batch_users = idxlist[batch_idx:end_idx]
            x = torch.FloatTensor(self.train_matrix[batch_users].toarray()).to(self.device)
            
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(x)
            anneal = min(0.2, epoch / 100)
            loss = loss_fn(recon_batch, x, mu, logvar, anneal)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / (self.num_users / batch_size)

    def evaluate(self, batch_size):
        self.model.eval()
        train_dense = self.train_matrix.toarray()
        recon_val_list = []
        
        with torch.no_grad():
            for i in range(0, self.num_users, batch_size):
                end_i = min(i + batch_size, self.num_users)
                batch_x = torch.FloatTensor(train_dense[i:end_i]).to(self.device)
                recon_batch, _, _ = self.model(batch_x)
                
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[train_dense[i:end_i] > 0] = -1e9
                recon_val_list.append(recon_batch)
            
            recon_val = np.concatenate(recon_val_list, axis=0)
            recall_10 = get_recall(recon_val, self.val_matrix.tocsr(), k=10)
            recall_100 = get_recall(recon_val, self.val_matrix.tocsr(), k=100)
        return recall_10, recall_100