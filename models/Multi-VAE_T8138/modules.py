import torch
import numpy as np

def loss_fn(recon_x, x, mu, logvar, anneal=1.0):
    log_softmax_var = torch.log_softmax(recon_x, dim=1)
    neg_ll = -torch.mean(torch.sum(log_softmax_var * x, dim=1))
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return neg_ll + anneal * kld

def get_recall(recon_batch, test_matrix, k=100):
    indices = np.argpartition(-recon_batch, k, axis=1)[:, :k]
    recall_list = []
    
    for i in range(len(indices)):
        actual = test_matrix[i].indices
        if len(actual) == 0: continue
        hits = len(set(indices[i]) & set(actual))
        recall_list.append(hits / len(actual))
        
    return np.mean(recall_list) if len(recall_list) > 0 else 0