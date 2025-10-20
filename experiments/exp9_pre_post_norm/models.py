import torch
import torch.nn as nn
from models.layers import MultiHeadAttention
from models.components import MixtureOfExperts


class PreNormMoEBlock(nn.Module):
    """Original: Pre-normalization only"""
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss


class PrePostNormMoEBlock(nn.Module):
    """New: Pre and Post normalization"""
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)
        
        self.pre_norm1 = nn.RMSNorm(d_model)
        self.pre_norm2 = nn.RMSNorm(d_model)
        self.post_norm1 = nn.RMSNorm(d_model)
        self.post_norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.post_norm1(self.attention(self.pre_norm1(x)))
        x = x + self.dropout(attn_out)
        ff_out, aux_loss = self.feed_forward(self.pre_norm2(x))
        ff_out = self.post_norm2(ff_out)
        x = x + self.dropout(ff_out)
        return x, aux_loss


class SimpleMoEModel(nn.Module):
    """Minimal model for comparison"""
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, 
                 num_experts=8, top_k=2, dropout=0.1, use_pre_post_norm=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        BlockClass = PrePostNormMoEBlock if use_pre_post_norm else PreNormMoEBlock
        self.blocks = nn.ModuleList([
            BlockClass(d_model, n_heads, d_ff, max_seq_len, num_experts, top_k, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        aux_loss_total = 0
        for block in self.blocks:
            x, aux_loss = block(x)
            aux_loss_total += aux_loss
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, aux_loss_total / len(self.blocks)

