"""
DeepSeek Sparse Attention (DSA) Implementation

This module implements the sparse attention mechanism described in DeepSeek-V3.2-Exp paper.

Key Components:
1. Lightning Indexer: Computes index scores between query and key tokens
2. Fine-grained Token Selection: Selects top-k tokens based on index scores
3. Sparse Attention: Applies attention only on selected tokens

Reference: DeepSeek-V3.2-Exp paper (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from typing import Optional, Tuple


class LightningIndexer(nn.Module):
    """
    Lightning Indexer for DeepSeek Sparse Attention
    
    Computes index scores I_{t,s} = Σ w_{t,j} · ReLU(q_{t,j} · k_s)
    
    Args:
        d_model: Model dimension
        indexer_heads: Number of indexer heads (H_I)
        indexer_dim: Dimension of indexer queries/keys (d_I)
        dropout: Dropout probability
    """
    def __init__(
        self, 
        d_model: int, 
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        
        # Indexer query projection: h_t -> {q_{t,j}^I}
        self.q_proj = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)
        
        # Indexer key projection: h_s -> k_s^I
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        
        # Indexer weights: w_{t,j}^I for each head
        self.w_proj = nn.Linear(d_model, indexer_heads, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute index scores between all pairs of tokens
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            index_scores: Index scores [batch_size, seq_len, seq_len]
                         index_scores[b, t, s] = I_{t,s}
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute indexer queries: [batch, seq_len, indexer_heads, indexer_dim]
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        
        # Compute indexer keys: [batch, seq_len, indexer_dim]
        keys = self.k_proj(x)
        
        # Compute indexer weights: [batch, seq_len, indexer_heads]
        weights = self.w_proj(x)
        
        # Compute dot products: q_{t,j} · k_s for all t, s, j
        # queries: [batch, seq_len_q, heads, dim]
        # keys: [batch, seq_len_k, dim]
        # Result: [batch, seq_len_q, heads, seq_len_k]
        dots = torch.einsum('bthd,bsd->bths', queries, keys)
        
        # Apply ReLU activation (for throughput efficiency as per paper)
        activated = F.relu(dots)  # [batch, seq_len_q, heads, seq_len_k]
        
        # Weight each head: w_{t,j} · ReLU(q_{t,j} · k_s)
        # weights: [batch, seq_len_q, heads] -> [batch, seq_len_q, heads, 1]
        # activated: [batch, seq_len_q, heads, seq_len_k]
        weighted = activated * weights.unsqueeze(-1)
        
        # Sum across heads: Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)
        # weighted: [batch, seq_len_q, heads, seq_len_k]
        index_scores = weighted.sum(dim=2)  # [batch, seq_len_q, seq_len_k]
        
        return index_scores


class TopKTokenSelector(nn.Module):
    """
    Fine-grained Token Selection Mechanism
    
    Selects top-k tokens based on index scores for each query token.
    
    Args:
        top_k: Number of tokens to select for each query
    """
    def __init__(self, top_k: int = 2048):
        super().__init__()
        self.top_k = top_k
        
    def forward(
        self, 
        index_scores: torch.Tensor,
        apply_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k tokens based on index scores
        
        Args:
            index_scores: Index scores [batch, seq_len_q, seq_len_k]
            apply_causal_mask: Whether to apply causal masking
            
        Returns:
            - top_k_mask: Boolean mask [batch, seq_len_q, seq_len_k]
                         True for selected tokens
            - top_k_indices: Indices of selected tokens [batch, seq_len_q, top_k]
        """
        batch_size, seq_len_q, seq_len_k = index_scores.shape
        
        # Apply causal mask: token t can only attend to tokens <= t
        if apply_causal_mask:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=index_scores.device),
                diagonal=1
            ).bool()
            # Set future tokens to very negative value
            index_scores = index_scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
        
        # Select top-k indices for each query token
        # Note: top_k is clamped to not exceed available tokens
        actual_k = min(self.top_k, seq_len_k)
        top_k_values, top_k_indices = torch.topk(
            index_scores, 
            k=actual_k, 
            dim=-1,
            largest=True
        )
        
        # Create boolean mask from indices
        top_k_mask = torch.zeros_like(index_scores, dtype=torch.bool)
        top_k_mask.scatter_(2, top_k_indices, True)
        
        return top_k_mask, top_k_indices


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA)
    
    Combines lightning indexer and top-k selection with standard attention mechanism.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        indexer_heads: Number of indexer heads
        indexer_dim: Dimension of indexer queries/keys
        sparse_top_k: Number of tokens to select (k in top-k)
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512,  # Adjusted for shorter sequences
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sparse_top_k = sparse_top_k
        
        # Main attention components
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryPositionalEmbeddings(dim=self.d_k, max_seq_len=max_seq_len, base=10000)
        self.dropout = dropout
        
        # Lightning indexer
        self.indexer = LightningIndexer(
            d_model=d_model,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            dropout=dropout
        )
        
        # Token selector
        self.selector = TopKTokenSelector(top_k=sparse_top_k)
        
        # Whether to use sparse attention (can be disabled during warmup)
        self.use_sparse = True
        
    def forward(
        self, 
        x: torch.Tensor,
        return_index_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sparse attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_index_scores: Whether to return index scores for analysis
            
        Returns:
            - output: Attention output [batch_size, seq_len, d_model]
            - index_scores: Index scores if return_index_scores=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, d_k]
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: [batch, heads, seq_len, d_k]
        
        # Apply RoPE
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)
        
        # Compute index scores with lightning indexer
        index_scores = self.indexer(x)  # [batch, seq_len, seq_len]
        
        if self.use_sparse:
            # Select top-k tokens
            top_k_mask, top_k_indices = self.selector(index_scores, apply_causal_mask=True)
            
            # Apply sparse attention using the mask
            # Create attention mask: -inf for non-selected tokens
            # Need to unsqueeze for broadcasting with heads dimension
            attn_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len,
                device=x.device,
                dtype=Q.dtype
            )
            attn_mask = attn_mask.masked_fill(~top_k_mask.unsqueeze(1), float('-inf'))
            
            # Apply attention with sparse mask
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Dense attention (for warmup stage)
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)
        
        if return_index_scores:
            return output, index_scores
        return output, None
    
    def enable_sparse(self):
        """Enable sparse attention (top-k selection)"""
        self.use_sparse = True
        
    def disable_sparse(self):
        """Disable sparse attention (use dense attention for warmup)"""
        self.use_sparse = False
    
    def get_indexer_parameters(self):
        """Get parameters of the lightning indexer only"""
        return self.indexer.parameters()
    
    def freeze_main_attention(self):
        """Freeze main attention parameters (for indexer warmup)"""
        for param in self.qkv.parameters():
            param.requires_grad = False
        for param in self.w_o.parameters():
            param.requires_grad = False
            
    def unfreeze_main_attention(self):
        """Unfreeze main attention parameters (for full training)"""
        for param in self.qkv.parameters():
            param.requires_grad = True
        for param in self.w_o.parameters():
            param.requires_grad = True


class SparseAttentionMetrics:
    """
    Utility class for computing metrics related to sparse attention
    """
    
    @staticmethod
    def compute_sparsity(top_k_mask: torch.Tensor) -> float:
        """
        Compute the sparsity ratio of the attention pattern
        
        Args:
            top_k_mask: Boolean mask [batch, seq_len, seq_len]
            
        Returns:
            sparsity: Fraction of zeros (1.0 = fully sparse, 0.0 = dense)
        """
        total_elements = top_k_mask.numel()
        selected_elements = top_k_mask.sum().item()
        sparsity = 1.0 - (selected_elements / total_elements)
        return sparsity
    
    @staticmethod
    def compute_indexer_alignment_loss(
        index_scores: torch.Tensor,
        attention_weights: torch.Tensor,
        selected_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence loss to align indexer with main attention
        
        This is the L_I loss from the paper:
        - Dense warmup: L_I = Σ_t KL(p_{t,:} || Softmax(I_{t,:}))
        - Sparse training: L_I = Σ_t KL(p_{t,S_t} || Softmax(I_{t,S_t}))
        
        Args:
            index_scores: Index scores [batch, seq_len, seq_len]
            attention_weights: Main attention weights [batch, seq_len, seq_len]
            selected_mask: Optional mask for selected tokens (for sparse stage)
            
        Returns:
            kl_loss: KL divergence loss
        """
        # Normalize attention weights (target distribution)
        # Sum across all attention heads if needed
        if attention_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
            attention_weights = attention_weights.sum(dim=1)  # Sum across heads
        
        # L1 normalize along sequence dimension
        target_dist = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Compute indexer distribution
        if selected_mask is not None:
            # Sparse stage: only consider selected tokens
            index_scores_masked = index_scores.masked_fill(~selected_mask, float('-inf'))
            indexer_dist = F.softmax(index_scores_masked, dim=-1)
            target_dist_masked = target_dist.masked_fill(~selected_mask, 0.0)
            target_dist_masked = target_dist_masked / (target_dist_masked.sum(dim=-1, keepdim=True) + 1e-9)
            target_dist = target_dist_masked
        else:
            # Dense stage: use all tokens
            indexer_dist = F.softmax(index_scores, dim=-1)
        
        # Compute KL divergence: KL(target || indexer)
        kl_loss = F.kl_div(
            indexer_dist.log(),
            target_dist,
            reduction='batchmean',
            log_target=False
        )
        
        return kl_loss
    
    @staticmethod
    def analyze_attention_pattern(
        top_k_indices: torch.Tensor,
        seq_len: int
    ) -> dict:
        """
        Analyze the attention pattern statistics
        
        Args:
            top_k_indices: Selected token indices [batch, seq_len, top_k]
            seq_len: Sequence length
            
        Returns:
            stats: Dictionary with pattern statistics
        """
        # Compute average distance to selected tokens
        positions = torch.arange(seq_len, device=top_k_indices.device).unsqueeze(0).unsqueeze(2)
        distances = (positions - top_k_indices).abs().float()
        
        stats = {
            'avg_distance': distances.mean().item(),
            'max_distance': distances.max().item(),
            'min_distance': distances.min().item(),
            'std_distance': distances.std().item(),
        }
        
        return stats
