import torch
import torch.nn as nn
import numpy as np

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return outputs.transpose(-1, -2)
        
class SASRec(torch.nn.Module):
    def __init__(self, item_num, config):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.dev = config.DEVICE
        
        self.item_emb = torch.nn.Embedding(self.item_num + 1, config.HIDDEN_UNITS, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(config.MAX_LEN, config.HIDDEN_UNITS)
        self.emb_dropout = torch.nn.Dropout(p=config.DROPOUT)
        
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        self.last_layernorm = torch.nn.LayerNorm(config.HIDDEN_UNITS, eps=1e-8)
        
        for _ in range(config.NUM_BLOCKS):
            self.attention_layernorms.append(torch.nn.LayerNorm(config.HIDDEN_UNITS, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(config.HIDDEN_UNITS, config.NUM_HEADS, dropout=config.DROPOUT)
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(config.HIDDEN_UNITS, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(config.HIDDEN_UNITS, config.DROPOUT))

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)
        
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q.transpose(0, 1), Q.transpose(0, 1), Q.transpose(0, 1), attn_mask=attention_mask)
            
            seqs = Q + mha_outputs.transpose(0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = seqs + self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(user_seqs)
        
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits

    def predict(self, user_seqs, item_indices):
        log_feats = self.log2feats(user_seqs) 
        final_feat = log_feats[:, -1, :] 
        item_embs = self.item_emb(item_indices) 
        scores = torch.matmul(final_feat, item_embs.transpose(0, 1)) 
        return scores
