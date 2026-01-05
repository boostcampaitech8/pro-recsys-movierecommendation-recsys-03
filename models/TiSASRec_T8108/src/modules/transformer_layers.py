"""
Time-Aware Transformer Layers for TiSASRec
논문: Time Interval Aware Self-Attention for Sequential Recommendation (WSDM 2020)

핵심 수식:
- Eq.6: z_i = sum_j(alpha_ij * (V_j + r^v_ij + p^v_j))
- Eq.8: e_ij = Q_i * (K_j + r^k_ij + p^k_j)^T / sqrt(d)

여기서:
- r^k_ij, r^v_ij: 상대 시간 간격 임베딩 (Key, Value용)
- p^k_j, p^v_j: 절대 위치 임베딩 (Key, Value용)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TiSASRecAttention(nn.Module):
    """
    논문의 Time Interval-Aware Self-attention Layer 정확 구현
    
    Eq.6: z_i = sum_j(alpha_ij * (m_sj * W^V + r^v_ij + p^v_j))
    Eq.8: e_ij = (m_si * W^Q) * (m_sj * W^K + r^k_ij + p^k_j)^T / sqrt(d)
    """
    
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})")
        
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // n_heads
        self.scale = math.sqrt(self.head_size)
        
        # W^Q, W^K, W^V projections
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        # Dropout and LayerNorm
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_k: torch.Tensor,
        pos_v: torch.Tensor,
        time_k: torch.Tensor,
        time_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) - 아이템 임베딩
            attention_mask: (batch, 1, seq_len, seq_len) - causal + padding mask
            pos_k: (seq_len, hidden_size) - 절대 위치 임베딩 for Key
            pos_v: (seq_len, hidden_size) - 절대 위치 임베딩 for Value
            time_k: (batch, seq_len, seq_len, hidden_size) - 시간 간격 임베딩 for Key
            time_v: (batch, seq_len, seq_len, hidden_size) - 시간 간격 임베딩 for Value
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Q, K, V 계산: (batch, seq_len, hidden_size)
        Q = self.W_Q(hidden_states)
        K = self.W_K(hidden_states)
        V = self.W_V(hidden_states)
        
        # Multi-head reshape: (batch, n_heads, seq_len, head_size)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        
        # ========== Eq.8: e_ij = Q_i * (K_j + r^k_ij + p^k_j)^T / sqrt(d) ==========
        
        # 1) Q * K^T: (batch, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))
        
        # 2) Q * p^k_j: 절대 위치 임베딩 (공유)
        # pos_k: (seq_len, hidden_size) -> (seq_len, n_heads, head_size)
        pos_k_heads = pos_k.view(seq_len, self.n_heads, self.head_size)
        # (batch, n_heads, seq_len, head_size) @ (n_heads, head_size, seq_len)
        # -> (batch, n_heads, seq_len, seq_len)
        pos_k_heads = pos_k_heads.permute(1, 2, 0)  # (n_heads, head_size, seq_len)
        pos_scores = torch.matmul(Q, pos_k_heads.unsqueeze(0))
        
        # 3) Q * r^k_ij: 상대 시간 간격 임베딩
        # time_k: (batch, seq_len, seq_len, hidden_size)
        # -> (batch, seq_len, seq_len, n_heads, head_size)
        time_k_heads = time_k.view(batch_size, seq_len, seq_len, self.n_heads, self.head_size)
        # -> (batch, n_heads, seq_len, seq_len, head_size)
        time_k_heads = time_k_heads.permute(0, 3, 1, 2, 4)
        # Q: (batch, n_heads, seq_len, 1, head_size)
        # time_k_heads: (batch, n_heads, seq_len, seq_len, head_size)
        # einsum으로 계산: Q_i * r^k_ij
        time_scores = torch.einsum('bnid,bnijd->bnij', Q, time_k_heads)
        
        # 합산
        attn_scores = (attn_scores + pos_scores + time_scores) / self.scale
        
        # Mask 적용
        attn_scores = attn_scores + attention_mask
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # ========== Eq.6: z_i = sum_j(alpha_ij * (V_j + r^v_ij + p^v_j)) ==========
        
        # 1) alpha * V: (batch, n_heads, seq_len, head_size)
        context = torch.matmul(attn_probs, V)
        
        # 2) alpha * p^v_j: 절대 위치
        pos_v_heads = pos_v.view(seq_len, self.n_heads, self.head_size)
        pos_v_heads = pos_v_heads.permute(1, 0, 2)  # (n_heads, seq_len, head_size)
        # (batch, n_heads, seq_len, seq_len) @ (n_heads, seq_len, head_size)
        pos_context = torch.matmul(attn_probs, pos_v_heads.unsqueeze(0))
        
        # 3) alpha * r^v_ij: 상대 시간 간격
        time_v_heads = time_v.view(batch_size, seq_len, seq_len, self.n_heads, self.head_size)
        time_v_heads = time_v_heads.permute(0, 3, 1, 2, 4)
        # einsum: alpha_ij * r^v_ij
        time_context = torch.einsum('bnij,bnijd->bnid', attn_probs, time_v_heads)
        
        # 합산
        context = context + pos_context + time_context
        
        # Reshape back: (batch, seq_len, hidden_size)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection + residual + layer norm
        output = self.dense(context)
        output = self.out_dropout(output)
        output = self.layer_norm(output + hidden_states)
        
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (Eq.9)"""
    
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        if hidden_act == 'gelu':
            self.activation = nn.GELU()
        elif hidden_act == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden = self.dense_1(input_tensor)
        hidden = self.activation(hidden)
        hidden = self.dense_2(hidden)
        hidden = self.dropout(hidden)
        return self.layer_norm(hidden + input_tensor)


class TiSASRecLayer(nn.Module):
    """Single TiSASRec Transformer Layer"""
    
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        
        self.attention = TiSASRecAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_k: torch.Tensor,
        pos_v: torch.Tensor,
        time_k: torch.Tensor,
        time_v: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.attention(hidden_states, attention_mask, pos_k, pos_v, time_k, time_v)
        output = self.ffn(attn_output)
        return output


class TiSASRecEncoder(nn.Module):
    """TiSASRec Transformer Encoder (Stack of Layers)"""
    
    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 2,
        hidden_size: int = 64,
        inner_size: int = 256,
        hidden_dropout_prob: float = 0.5,
        attn_dropout_prob: float = 0.5,
        hidden_act: str = 'gelu',
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TiSASRecLayer(
                n_heads=n_heads,
                hidden_size=hidden_size,
                inner_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(n_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_k: torch.Tensor,
        pos_v: torch.Tensor,
        time_k: torch.Tensor,
        time_v: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ):
        all_encoder_layers = []
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, pos_k, pos_v, time_k, time_v)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers