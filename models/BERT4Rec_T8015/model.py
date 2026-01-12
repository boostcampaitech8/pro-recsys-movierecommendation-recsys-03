import torch
import torch.nn as nn
import numpy as np

class BERT4Rec(nn.Module):
    def __init__(self, item_num, config):
        super(BERT4Rec, self).__init__()
        self.item_num = item_num
        self.dev = config.DEVICE
        
        self.item_emb = nn.Embedding(self.item_num + 2, config.HIDDEN_UNITS, padding_idx=0)
        
        self.pos_emb = nn.Embedding(config.MAX_LEN, config.HIDDEN_UNITS)
        self.emb_dropout = nn.Dropout(p=config.DROPOUT)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.HIDDEN_UNITS,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.HIDDEN_UNITS * 4,
            dropout=config.DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_BLOCKS)
        
        self.out_layer = nn.Linear(config.HIDDEN_UNITS, self.item_num + 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def log2feats(self, log_seqs):
        seq_len = log_seqs.shape[1]
        
        positions = torch.arange(seq_len, device=self.dev).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        
        seqs = self.item_emb(log_seqs) + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        
        key_padding_mask = (log_seqs == 0)
        log_feats = self.transformer_encoder(seqs, src_key_padding_mask=key_padding_mask)
        
        return log_feats

    def forward(self, log_seqs):
        log_feats = self.log2feats(log_seqs)
        logits = self.out_layer(log_feats)
        
        return logits

    def predict(self, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :] 
        
        logits = self.out_layer(final_feat)
        
        if item_indices is not None:
            return logits[:, item_indices]
        
        return logits
