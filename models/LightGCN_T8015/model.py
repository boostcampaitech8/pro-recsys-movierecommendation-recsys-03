import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_total_nodes, config, adj_matrix):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_total_nodes = num_total_nodes
        self.config = config
        self.adj_matrix = adj_matrix
        
        self.entity_embedding = nn.Embedding(num_total_nodes, config.embedding_dim)
        
        nn.init.normal_(self.entity_embedding.weight, std=0.1)
        
    def forward(self):
        all_embeddings = self.entity_embedding.weight
        embeddings_list = [all_embeddings]
        
        for _ in range(self.config.num_layers):
            all_embeddings = torch.sparse.mm(self.adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
            
        # Shape: (K+1, num_total_nodes, embedding_dim)
        embeddings_list = torch.stack(embeddings_list, dim=0)
        light_out = torch.mean(embeddings_list, dim=0)
        
        final_user_embeddings = light_out[:self.num_users]
        final_item_embeddings = light_out[self.num_users : self.num_users + self.num_items]
        
        return final_user_embeddings, final_item_embeddings
        
    def bpr_loss(self, users, pos_items, neg_items, confidence):
        with torch.cuda.amp.autocast(enabled=False):
            all_user_emb, all_item_emb = self.forward()
        
        u_emb = all_user_emb[users]
        pos_i_emb = all_item_emb[pos_items]
        neg_i_emb = all_item_emb[neg_items]
        
        pos_scores = torch.mul(u_emb, pos_i_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_i_emb).sum(dim=1)
        
        # BPR Loss
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores) * confidence)
        
        u_emb_0 = self.entity_embedding(users)
        
        pos_items_shifted = pos_items + self.num_users
        neg_items_shifted = neg_items + self.num_users
        
        pos_i_emb_0 = self.entity_embedding(pos_items_shifted)
        neg_i_emb_0 = self.entity_embedding(neg_items_shifted)
        
        reg_loss = (u_emb_0**2).sum() + (pos_i_emb_0**2).sum() + (neg_i_emb_0**2).sum()
        reg_loss = 0.5 * reg_loss / len(users)
        
        total_loss = loss + self.config.decay * reg_loss
        
        return total_loss, loss, self.config.decay * reg_loss

    def predict(self, users):
        final_user_emb, final_item_emb = self.forward()
        u_emb = final_user_emb[users]
        scores = torch.mm(u_emb, final_item_emb.t())
        return scores
