"""
Attention Mechanism Implementations for Ablation Study

This module implements various attention mechanisms with a unified interface
for fair comparison in the ablation study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torchtune.modules import RotaryPositionalEmbeddings


class Rotary(nn.Module):
    """Rotary Positional Embeddings wrapper"""
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=10000)

    def forward(self, x_BTHD: torch.Tensor):
        return self.rope(x_BTHD)


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (MHA)
    
    The baseline attention mechanism used in vanilla Transformers.
    Each head has independent Q, K, V projections.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        # Apply RoPE
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    
    Uses low-rank compression for KV cache to reduce memory.
    From DeepSeek-V2/V3 architecture.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        qk_rope_dim: int,
        qk_nope_dim: int,
        kv_lora_rank: int,
        v_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qk_dim = qk_rope_dim + qk_nope_dim
        self.qk_rope_dim, self.qk_nope_dim = qk_rope_dim, qk_nope_dim
        self.kv_lora_dim = kv_lora_rank
        self.v_dim = v_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.query = nn.Linear(d_model, n_heads * self.qk_dim, bias=False)
        self.compressed_kv = nn.Linear(d_model, kv_lora_rank + qk_rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        self.decompressed_kv = nn.Linear(
            kv_lora_rank, n_heads * (qk_nope_dim + v_dim), bias=False
        )
        self.w_o = nn.Linear(v_dim * n_heads, d_model, bias=False)
        self.rotary = Rotary(qk_rope_dim, max_seq_len)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.size(0), x.size(1)

        # Query part
        q = self.query.forward(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_dim)
        q_nope, q_rope = torch.split(q, (self.qk_nope_dim, self.qk_rope_dim), dim=-1)
        q_rope = self.rotary.forward(q_rope)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # KV part with compression
        kv = self.compressed_kv.forward(x)
        kv, k_rope = torch.split(kv, (self.kv_lora_dim, self.qk_rope_dim), dim=-1)
        
        # K rope part
        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_dim)
        k_rope = self.rotary.forward(k_rope)
        
        # V and K nope part
        kv = self.kv_norm.forward(kv)
        kv = self.decompressed_kv.forward(kv)
        kv = kv.view(batch_size, seq_len, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope, v = torch.split(kv, (self.qk_nope_dim, self.v_dim), dim=-1)
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_heads, -1)], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        return self.w_o.forward(attn_output)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    
    Shares K and V across groups of Q heads to reduce parameters
    and KV cache size. Middle ground between MHA and MQA.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_rep = n_heads // n_kv_heads  # Number of Q heads per KV head

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Project Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # Apply RoPE
        Q = self.rotary(Q)
        K = self.rotary(K)
        
        # Repeat K and V for each group
        K = K.repeat_interleave(self.n_rep, dim=2)  # [B, T, H, D]
        V = V.repeat_interleave(self.n_rep, dim=2)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # [B, H, T, D]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA)
    
    Single shared K and V across all Q heads.
    Extreme parameter and KV cache reduction.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_k, bias=False)  # Single K
        self.v_proj = nn.Linear(d_model, self.d_k, bias=False)  # Single V
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Project Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.k_proj(x).view(batch_size, seq_len, 1, self.d_k)
        V = self.v_proj(x).view(batch_size, seq_len, 1, self.d_k)
        
        # Apply RoPE
        Q = self.rotary(Q)
        K = self.rotary(K)
        
        # Expand K and V to match number of heads
        K = K.expand(-1, -1, self.n_heads, -1)
        V = V.expand(-1, -1, self.n_heads, -1)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # [B, H, T, D]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention
    
    Each token only attends to a fixed window of previous tokens.
    Reduces computation from O(n²) to O(n·w) where w is window size.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        window_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        assert d_model % n_heads == 0

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        # Apply RoPE
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        # Create sliding window mask
        # For each position i, can only attend to [max(0, i-window_size), i]
        attn_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1)  # Causal mask
        
        # Add window constraint
        for i in range(seq_len):
            if i >= self.window_size:
                attn_mask[i, :i-self.window_size] = True
        
        # Convert to additive mask
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek-style Sparse Attention
    
    Uses learned indexer to select top-k most relevant tokens.
    Adaptive sparsity based on content rather than fixed patterns.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sparse_top_k = sparse_top_k
        
        assert d_model % n_heads == 0

        # Main attention
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout
        
        # Lightning indexer for token selection
        self.indexer_q = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)
        self.indexer_k = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Compute Q, K, V for main attention
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]
        
        # Apply RoPE
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)
        
        # Compute indexer scores
        iq = self.indexer_q(x).view(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        ik = self.indexer_k(x).view(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        
        # Compute scores: [B, T, T]
        index_scores = torch.einsum('bqhd,bkhd->bqk', iq, ik) / (self.indexer_dim ** 0.5)
        
        # Apply causal mask and select top-k
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        index_scores = index_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
        
        # Get top-k indices per query
        k = min(self.sparse_top_k, seq_len)
        _, top_k_indices = torch.topk(index_scores, k, dim=-1)  # [B, T, k]
        
        # Create sparse attention mask
        attn_mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device, dtype=torch.bool)
        for b in range(batch_size):
            for q in range(seq_len):
                attn_mask[b, q, top_k_indices[b, q]] = True
        
        # Apply causal constraint
        attn_mask = attn_mask & (~causal_mask.unsqueeze(0))
        
        # Convert to additive mask
        attn_mask = attn_mask.unsqueeze(1)  # [B, 1, T, T] for broadcasting over heads
        attn_mask = (~attn_mask).masked_fill(~attn_mask, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class IdentityAttention(nn.Module):
    """
    Identity/No Attention
    
    Simply passes input through without any attention computation.
    Tests whether attention is necessary at all layers.
    """
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        # Simple linear projection to maintain parameter count similarity
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.projection(x)


# Linear attention via FLA will be imported separately since it requires external dependency

