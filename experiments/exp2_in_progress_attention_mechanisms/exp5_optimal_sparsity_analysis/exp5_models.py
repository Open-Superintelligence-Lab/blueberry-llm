"""
Model Definitions for Experiment 5: Optimal Sparsity Analysis

This module defines the sparse attention model with configurable sparsity ratios
for systematic analysis of optimal sparsity levels across sequence lengths.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

# Import from parent models package
from models.components import MixtureOfExperts
from models.layers import MultiHeadAttention

# Import sparse attention from exp1
sys.path.insert(0, os.path.join(root_dir, 'experiments', 'exp1_sparse_vs_classic_attention'))
from sparse_attention import DeepSeekSparseAttention


class ConfigurableSparseTransformerBlock(nn.Module):
    """
    Transformer block with configurable DeepSeek Sparse Attention
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # DeepSeek Sparse Attention with configurable top_k
        self.attention = DeepSeekSparseAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            sparse_top_k=sparse_top_k,
            dropout=dropout
        )
        
        # MoE layer
        self.feed_forward = MixtureOfExperts(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=expert_top_k,
            dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        return_index_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass
        
        Returns:
            - output: Block output
            - aux_loss: MoE auxiliary loss
            - index_scores: Index scores if return_index_scores=True
        """
        # Self-attention with sparse attention
        attn_out, index_scores = self.attention(
            self.norm1(x), 
            return_index_scores=return_index_scores
        )
        x = x + self.dropout(attn_out)
        
        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, aux_loss, index_scores
    
    def update_sparsity(self, new_sparse_top_k: int):
        """Update the sparsity level by changing top_k"""
        self.attention.sparse_top_k = new_sparse_top_k
        self.attention.selector.top_k = new_sparse_top_k


class ConfigurableSparseAttentionMoELLM(nn.Module):
    """
    Language Model with configurable DeepSeek Sparse Attention and MoE
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.load_balancing_weight = load_balancing_weight
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks with configurable sparse attention
        self.blocks = nn.ModuleList([
            ConfigurableSparseTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                num_experts=num_experts,
                expert_top_k=expert_top_k,
                indexer_heads=indexer_heads,
                indexer_dim=indexer_dim,
                sparse_top_k=sparse_top_k,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
    def forward(
        self, 
        x: torch.Tensor,
        return_index_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Forward pass
        
        Args:
            x: Input token indices [batch_size, seq_len]
            return_index_scores: Whether to return index scores
            
        Returns:
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - total_aux_loss: Total auxiliary loss (MoE load balancing)
            - index_scores_list: List of index scores per layer (if requested)
        """
        # Embed tokens
        x = self.embed(x)
        
        # Pass through transformer blocks
        total_aux_loss = 0.0
        index_scores_list = [] if return_index_scores else None
        
        for block in self.blocks:
            x, aux_loss, index_scores = block(x, return_index_scores=return_index_scores)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
            if return_index_scores and index_scores is not None:
                index_scores_list.append(index_scores)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Scale auxiliary loss
        if total_aux_loss != 0.0:
            total_aux_loss = total_aux_loss * self.load_balancing_weight
        else:
            total_aux_loss = None
            
        return logits, total_aux_loss, index_scores_list
    
    def update_sparsity(self, new_sparse_top_k: int):
        """Update sparsity level for all blocks"""
        for block in self.blocks:
            block.update_sparsity(new_sparse_top_k)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_sparse_model(config: dict) -> ConfigurableSparseAttentionMoELLM:
    """
    Create configurable sparse attention model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: ConfigurableSparseAttentionMoELLM instance
    """
    model = ConfigurableSparseAttentionMoELLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        num_experts=config.get('num_experts', 4),
        expert_top_k=config.get('expert_top_k', 2),
        indexer_heads=config.get('indexer_heads', 4),
        indexer_dim=config.get('indexer_dim', 64),
        sparse_top_k=config.get('sparse_top_k', 512),
        dropout=config.get('dropout', 0.1),
        load_balancing_weight=config.get('load_balancing_weight', 0.01)
    )
    return model


def get_sparse_top_k(seq_len: int, sparsity_ratio: float) -> int:
    """
    Calculate top-k based on sparsity ratio
    
    Args:
        seq_len: Sequence length
        sparsity_ratio: Desired sparsity ratio (0.0 = fully dense, 1.0 = fully sparse)
        
    Returns:
        top_k: Number of tokens to select (at least 1)
    """
    return max(1, int(seq_len * (1.0 - sparsity_ratio)))
