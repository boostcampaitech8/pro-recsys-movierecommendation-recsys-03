"""
TiSASRec Model - 논문 정확 구현
Time Interval Aware Self-Attention for Sequential Recommendation (WSDM 2020)

핵심 구현:
1. Personalized Time Intervals (Section 3.2)
   - r_ij = |t_i - t_j| / r_min (유저별 최소 시간 간격으로 나눔)
   - clip(r_ij, max=k)
   
2. Separate Embeddings for K and V (Section 3.3)
   - M^P_K, M^P_V: 절대 위치 임베딩 (Key, Value용 각각)
   - M^R_K, M^R_V: 상대 시간 간격 임베딩 (Key, Value용 각각)
   
3. Time-Aware Self-Attention (Section 3.4)
   - Eq.6: z_i = sum_j(alpha_ij * (V_j + r^v_ij + p^v_j))
   - Eq.8: e_ij = Q_i * (K_j + r^k_ij + p^k_j)^T / sqrt(d)
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.transformer_layers import TiSASRecEncoder


class TiSASRec(SequentialRecommender):
    """TiSASRec: Time Interval aware Self-Attention Sequential Recommendation"""
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # Config
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        
        # TiSASRec 전용: max time interval (k in paper)
        self.max_time_interval = config['max_time_interval'] if 'max_time_interval' in config.final_config_dict else self.max_seq_length
        
        # 시간 필드
        self.TIME_FIELD = config['TIME_FIELD']
        self.timestamp_list = self.TIME_FIELD + '_list'
        
        # Loss
        self.loss_type = config['loss_type']
        
        # ============== Embeddings ==============
        # Item embedding: M^I
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        
        # 논문 Section 3.3: Key/Value용 별도 위치 임베딩
        # M^P_K, M^P_V: (max_seq_len, hidden_size)
        self.position_embedding_k = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.position_embedding_v = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # M^R_K, M^R_V: (max_time_interval, hidden_size)
        # 시간 간격 0 ~ k까지 커버
        self.time_interval_embedding_k = nn.Embedding(self.max_time_interval + 1, self.hidden_size)
        self.time_interval_embedding_v = nn.Embedding(self.max_time_interval + 1, self.hidden_size)
        
        # ============== Encoder ==============
        self.encoder = TiSASRecEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        
        # ============== Output ==============
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def compute_time_interval_matrix(self, time_seq):
        """
        논문 Section 3.2: Personalized Time Intervals
        
        Args:
            time_seq: (batch, seq_len) - 타임스탬프 시퀀스
        
        Returns:
            time_matrix: (batch, seq_len, seq_len) - 정규화된 시간 간격 인덱스
        """
        batch_size, seq_len = time_seq.size()
        
        # |t_i - t_j| 계산: (batch, seq_len, seq_len)
        time_diff = torch.abs(time_seq.unsqueeze(2) - time_seq.unsqueeze(1))
        
        # Personalized scaling: 유저별 최소 시간 간격으로 나눔
        # 0이 아닌 최소값 찾기 (각 배치별)
        # time_diff를 flatten하고 0을 큰 값으로 대체
        time_diff_flat = time_diff.view(batch_size, -1)
        
        # 0을 매우 큰 값으로 대체하여 min 계산에서 제외
        large_val = time_diff_flat.max() + 1
        time_diff_nonzero = time_diff_flat.clone()
        time_diff_nonzero[time_diff_nonzero == 0] = large_val
        
        # 유저별 최소 시간 간격: (batch,)
        r_min, _ = time_diff_nonzero.min(dim=1)
        r_min = r_min.clamp(min=1)  # 0으로 나누는 것 방지
        
        # Personalized scaling: r_ij = |t_i - t_j| / r_min
        # (batch, seq_len, seq_len) / (batch, 1, 1)
        time_matrix = time_diff / r_min.view(batch_size, 1, 1)
        
        # Clip to max_time_interval (k in paper)
        time_matrix = time_matrix.clamp(max=self.max_time_interval)
        
        # 정수 인덱스로 변환
        time_matrix = time_matrix.long()
        
        return time_matrix
    
    def get_attention_mask(self, item_seq):
        """Causal attention mask (lower triangular)"""
        attention_mask = (item_seq > 0).long()
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        max_len = attention_mask.size(-1)
        subsequent_mask = torch.triu(torch.ones((max_len, max_len), device=item_seq.device), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).long()
        
        extended_mask = extended_mask * subsequent_mask.unsqueeze(0).unsqueeze(0)
        extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        return extended_mask
    
    def forward(self, item_seq, item_seq_len, time_seq=None):
        """
        Forward pass
        
        Args:
            item_seq: (batch, seq_len) - 아이템 ID 시퀀스
            item_seq_len: (batch,) - 실제 시퀀스 길이
            time_seq: (batch, seq_len) - 타임스탬프 시퀀스
        """
        batch_size, seq_len = item_seq.size()
        device = item_seq.device
        
        # Item embedding: (batch, seq_len, hidden_size)
        item_emb = self.item_embedding(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)
        
        # Attention mask
        attention_mask = self.get_attention_mask(item_seq)
        
        # Position embeddings (shared across batch)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        pos_k = self.position_embedding_k(position_ids)  # (seq_len, hidden_size)
        pos_v = self.position_embedding_v(position_ids)  # (seq_len, hidden_size)
        
        # Time interval matrix and embeddings
        if time_seq is not None:
            time_matrix = self.compute_time_interval_matrix(time_seq)  # (batch, seq_len, seq_len)
            time_k = self.time_interval_embedding_k(time_matrix)  # (batch, seq_len, seq_len, hidden_size)
            time_v = self.time_interval_embedding_v(time_matrix)
        else:
            # 시간 정보가 없으면 0으로 설정
            time_matrix = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long, device=device)
            time_k = self.time_interval_embedding_k(time_matrix)
            time_v = self.time_interval_embedding_v(time_matrix)
        
        # Encoder
        encoder_outputs = self.encoder(
            item_emb,
            attention_mask,
            pos_k,
            pos_v,
            time_k,
            time_v,
            output_all_encoded_layers=True
        )
        
        return encoder_outputs[-1]
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # 시간 정보
        time_seq = interaction[self.timestamp_list] if self.timestamp_list in interaction else None
        
        # Forward
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        return loss
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        time_seq = interaction[self.timestamp_list] if self.timestamp_list in interaction else None
        
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        
        test_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_emb).sum(dim=1)
        
        return scores
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.timestamp_list] if self.timestamp_list in interaction else None
        
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        
        scores = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        
        return scores