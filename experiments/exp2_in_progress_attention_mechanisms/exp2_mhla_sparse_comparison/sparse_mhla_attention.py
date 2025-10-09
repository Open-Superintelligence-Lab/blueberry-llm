"""
DeepSeek Multi-Head Latent Attention (MHLA) with Sparse Attention

This module implements:
1. Lightning Indexer for sparse token selection
2. DeepSeek MHLA with sparse attention capability
3. Standard DeepSeek MHLA (dense baseline)

The sparse version adds top-k token selection to DeepSeek's latent attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

from models.deepseek_v3.deepseek_modeling import DeepseekV3Attention
from models.deepseek_v3.configuration_deepseek import DeepseekV3Config


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


class DeepSeekSparseMLHA(nn.Module):
    """
    DeepSeek Multi-Head Latent Attention with Sparse Attention
    
    Combines DeepSeek's MHLA with Lightning Indexer for sparse token selection.
    
    Args:
        config: DeepseekV3Config
        layer_idx: Layer index
        indexer_heads: Number of indexer heads
        indexer_dim: Dimension of indexer queries/keys
        sparse_top_k: Number of tokens to select (k in top-k)
    """
    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: Optional[int] = None,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512
    ):
        super().__init__()
        
        # DeepSeek MHLA (standard attention)
        self.mhla = DeepseekV3Attention(config, layer_idx=layer_idx)
        
        # Lightning indexer for sparse selection
        self.indexer = LightningIndexer(
            d_model=config.hidden_size,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            dropout=config.attention_dropout
        )
        
        # Token selector
        self.selector = TopKTokenSelector(top_k=sparse_top_k)
        
        # Whether to use sparse attention
        self.use_sparse = True
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        return_index_scores: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[any], Optional[torch.Tensor]]:
        """
        Forward pass with sparse MHLA
        
        Returns:
            - attn_output: Attention output
            - attn_weights: Attention weights (if output_attentions=True)
            - past_key_value: Cache (if use_cache=True)
            - index_scores: Index scores (if return_index_scores=True)
        """
        if self.use_sparse:
            # Compute index scores with lightning indexer
            index_scores = self.indexer(hidden_states)  # [batch, seq_len, seq_len]
            
            # Select top-k tokens
            top_k_mask, top_k_indices = self.selector(index_scores, apply_causal_mask=True)
            
            # Create sparse attention mask for MHLA
            # MHLA expects attention_mask in format: (batch, 1, seq_len, seq_len)
            # where 0 = attend, large negative = mask
            batch_size, seq_len, _ = hidden_states.shape
            sparse_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            # Mask non-selected tokens with large negative value
            sparse_mask = sparse_mask.masked_fill(~top_k_mask.unsqueeze(1), -1e9)
            
            # Combine with existing attention mask if present
            if attention_mask is not None:
                sparse_mask = sparse_mask + attention_mask
            
            # Forward through MHLA with sparse mask
            attn_output, attn_weights, past_key_value = self.mhla(
                hidden_states=hidden_states,
                attention_mask=sparse_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )
            
            if return_index_scores:
                return attn_output, attn_weights, past_key_value, index_scores
            else:
                return attn_output, attn_weights, past_key_value, None
        else:
            # Dense MHLA (no sparse selection)
            attn_output, attn_weights, past_key_value = self.mhla(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )
            return attn_output, attn_weights, past_key_value, None
    
    def enable_sparse(self):
        """Enable sparse attention (top-k selection)"""
        self.use_sparse = True
        
    def disable_sparse(self):
        """Disable sparse attention (use dense MHLA)"""
        self.use_sparse = False
    
    def get_indexer_parameters(self):
        """Get parameters of the lightning indexer only"""
        return self.indexer.parameters()
    
    def freeze_main_attention(self):
        """Freeze main MHLA parameters (for indexer warmup)"""
        for param in self.mhla.parameters():
            param.requires_grad = False
            
    def unfreeze_main_attention(self):
        """Unfreeze main MHLA parameters (for full training)"""
        for param in self.mhla.parameters():
            param.requires_grad = True

