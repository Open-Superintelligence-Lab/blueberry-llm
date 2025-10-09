"""
Efficient Attention Patterns for Experiment 4

This module implements various efficient attention patterns that combine
local and global attention mechanisms with sparse selection.

Patterns:
1. Local + Global: Local window + global sparse selection
2. Sliding Window: Overlapping windows with sparse selection
3. Hierarchical: Multi-scale token selection
4. Strided: Regular strided patterns with sparse selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class LocalGlobalPattern(nn.Module):
    """
    Local + Global attention pattern
    
    Combines local attention (within a window) with global sparse attention
    for better efficiency and quality.
    
    Args:
        local_window: Size of local attention window
        global_k: Number of global tokens to select
        indexer_dim: Dimension for global indexer
    """
    def __init__(
        self,
        local_window: int = 32,
        global_k: int = 64,
        indexer_dim: int = 32,
        d_model: int = 256
    ):
        super().__init__()
        self.local_window = local_window
        self.global_k = global_k
        self.indexer_dim = indexer_dim
        
        # Global indexer for selecting global tokens
        self.global_q_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.global_k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create local + global attention mask
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            attention_mask: Combined local + global mask [batch, 1, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Initialize attention mask (0 = attend, -inf = mask)
        attention_mask = torch.full(
            (batch_size, 1, seq_len, seq_len), 
            float('-inf'), 
            device=device, 
            dtype=x.dtype
        )
        
        # Local attention: attend to tokens within window
        for i in range(seq_len):
            start = max(0, i - self.local_window + 1)
            end = min(seq_len, i + 1)  # Causal: only attend to past tokens
            attention_mask[:, :, i, start:end] = 0
        
        # Global attention: select top-k global tokens using indexer
        if self.global_k > 0 and seq_len > self.local_window:
            global_queries = self.global_q_proj(x)  # [batch, seq_len, indexer_dim]
            global_keys = self.global_k_proj(x)     # [batch, seq_len, indexer_dim]
            
            # Compute global scores
            global_scores = torch.einsum('bqd,bkd->bqk', global_queries, global_keys)
            global_scores = F.relu(global_scores)
            
            # Select top-k global tokens for each query
            _, top_k_indices = torch.topk(global_scores, 
                                        min(self.global_k, seq_len), 
                                        dim=-1)
            
            # Add global attention to mask
            for b in range(batch_size):
                for q in range(seq_len):
                    global_tokens = top_k_indices[b, q]
                    attention_mask[b, :, q, global_tokens] = 0
        
        return attention_mask


class SlidingWindowPattern(nn.Module):
    """
    Sliding window attention with sparse selection
    
    Uses overlapping windows with sparse token selection within each window
    for better coverage and efficiency.
    
    Args:
        window_size: Size of each sliding window
        stride: Stride between windows
        sparse_ratio: Fraction of tokens to select within each window
    """
    def __init__(
        self,
        window_size: int = 64,
        stride: int = 32,
        sparse_ratio: float = 0.5,
        indexer_dim: int = 32,
        d_model: int = 256
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.sparse_ratio = sparse_ratio
        self.indexer_dim = indexer_dim
        
        # Indexer for selecting tokens within windows
        self.q_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sliding window attention mask with sparse selection
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            attention_mask: Sliding window + sparse mask [batch, 1, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Initialize mask
        attention_mask = torch.full(
            (batch_size, 1, seq_len, seq_len), 
            float('-inf'), 
            device=device, 
            dtype=x.dtype
        )
        
        # Compute indexer scores for sparse selection
        queries = self.q_proj(x)  # [batch, seq_len, indexer_dim]
        keys = self.k_proj(x)     # [batch, seq_len, indexer_dim]
        scores = torch.einsum('bqd,bkd->bqk', queries, keys)
        scores = F.relu(scores)
        
        # Apply sliding window with sparse selection
        for start in range(0, seq_len, self.stride):
            end = min(seq_len, start + self.window_size)
            
            # For each query position in this window
            for q in range(start, end):
                # Get key positions in this window (causal)
                key_start = max(0, q - self.window_size + 1)
                key_end = min(end, q + 1)
                
                if key_start < key_end:
                    # Get scores for this window
                    window_scores = scores[:, q, key_start:key_end]
                    
                    # Select top sparse_ratio tokens
                    k = max(1, int((key_end - key_start) * self.sparse_ratio))
                    _, top_k_indices = torch.topk(window_scores, k, dim=-1)
                    
                    # Add selected tokens to mask
                    for b in range(batch_size):
                        selected_indices = key_start + top_k_indices[b]
                        attention_mask[b, :, q, selected_indices] = 0
        
        return attention_mask


class HierarchicalPattern(nn.Module):
    """
    Hierarchical attention pattern with multi-scale selection
    
    Uses multiple scales of attention: local, medium-range, and global
    with different sparsity levels at each scale.
    
    Args:
        local_window: Local attention window size
        medium_window: Medium-range window size
        global_k: Number of global tokens
        local_sparse: Sparsity ratio for local attention
        medium_sparse: Sparsity ratio for medium-range attention
    """
    def __init__(
        self,
        local_window: int = 16,
        medium_window: int = 64,
        global_k: int = 32,
        local_sparse: float = 0.8,    # 80% sparse locally
        medium_sparse: float = 0.6,   # 60% sparse medium-range
        indexer_dim: int = 32,
        d_model: int = 256
    ):
        super().__init__()
        self.local_window = local_window
        self.medium_window = medium_window
        self.global_k = global_k
        self.local_sparse = local_sparse
        self.medium_sparse = medium_sparse
        self.indexer_dim = indexer_dim
        
        # Indexers for different scales
        self.local_indexer = nn.Linear(d_model, indexer_dim, bias=False)
        self.medium_indexer = nn.Linear(d_model, indexer_dim, bias=False)
        self.global_indexer = nn.Linear(d_model, indexer_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create hierarchical attention mask
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            attention_mask: Hierarchical attention mask [batch, 1, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Initialize mask
        attention_mask = torch.full(
            (batch_size, 1, seq_len, seq_len), 
            float('-inf'), 
            device=device, 
            dtype=x.dtype
        )
        
        # Compute indexer features
        local_features = self.local_indexer(x)
        medium_features = self.medium_indexer(x)
        global_features = self.global_indexer(x)
        
        for q in range(seq_len):
            # Local attention (high density)
            local_start = max(0, q - self.local_window + 1)
            local_end = q + 1
            
            if local_start < local_end:
                # Select top local_sparse tokens locally
                local_scores = torch.einsum('bd,bkd->bk', 
                                          local_features[:, q], 
                                          local_features[:, local_start:local_end])
                local_scores = F.relu(local_scores)
                
                k_local = max(1, int((local_end - local_start) * (1 - self.local_sparse)))
                _, top_k_local = torch.topk(local_scores, k_local, dim=-1)
                
                for b in range(batch_size):
                    selected_indices = local_start + top_k_local[b]
                    attention_mask[b, :, q, selected_indices] = 0
            
            # Medium-range attention (medium density)
            medium_start = max(0, q - self.medium_window + 1)
            medium_end = max(local_end, q + 1)
            
            if medium_start < medium_end and medium_end > local_end:
                # Select top medium_sparse tokens in medium range
                medium_scores = torch.einsum('bd,bkd->bk', 
                                           medium_features[:, q], 
                                           medium_features[:, medium_start:medium_end])
                medium_scores = F.relu(medium_scores)
                
                k_medium = max(1, int((medium_end - medium_start) * (1 - self.medium_sparse)))
                _, top_k_medium = torch.topk(medium_scores, k_medium, dim=-1)
                
                for b in range(batch_size):
                    selected_indices = medium_start + top_k_medium[b]
                    attention_mask[b, :, q, selected_indices] = 0
            
            # Global attention (low density)
            if self.global_k > 0 and seq_len > self.medium_window:
                global_scores = torch.einsum('bd,bkd->bk', 
                                           global_features[:, q], 
                                           global_features)
                global_scores = F.relu(global_scores)
                
                # Remove local and medium ranges from global selection
                global_scores[:, local_start:medium_end] = float('-inf')
                
                _, top_k_global = torch.topk(global_scores, self.global_k, dim=-1)
                
                for b in range(batch_size):
                    attention_mask[b, :, q, top_k_global[b]] = 0
        
        return attention_mask


class StridedPattern(nn.Module):
    """
    Strided attention pattern with sparse selection
    
    Uses regular strided patterns combined with sparse token selection
    for predictable and efficient attention patterns.
    
    Args:
        stride: Stride between attended positions
        sparse_ratio: Additional sparsity ratio within stride
        indexer_dim: Dimension for indexer
    """
    def __init__(
        self,
        stride: int = 4,
        sparse_ratio: float = 0.5,
        indexer_dim: int = 32,
        d_model: int = 256
    ):
        super().__init__()
        self.stride = stride
        self.sparse_ratio = sparse_ratio
        self.indexer_dim = indexer_dim
        
        # Indexer for sparse selection within stride
        self.q_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create strided attention mask with sparse selection
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            attention_mask: Strided + sparse mask [batch, 1, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Initialize mask
        attention_mask = torch.full(
            (batch_size, 1, seq_len, seq_len), 
            float('-inf'), 
            device=device, 
            dtype=x.dtype
        )
        
        # Compute indexer scores
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        scores = torch.einsum('bqd,bkd->bqk', queries, keys)
        scores = F.relu(scores)
        
        # Apply strided pattern with sparse selection
        for q in range(seq_len):
            # Get strided positions (causal)
            strided_positions = []
            for k in range(0, q + 1, self.stride):
                strided_positions.append(k)
            
            if strided_positions:
                # Select sparse_ratio of strided positions
                strided_scores = scores[:, q, strided_positions]
                k = max(1, int(len(strided_positions) * (1 - self.sparse_ratio)))
                _, top_k_indices = torch.topk(strided_scores, k, dim=-1)
                
                # Add selected positions to mask
                for b in range(batch_size):
                    selected_positions = [strided_positions[i] for i in top_k_indices[b]]
                    attention_mask[b, :, q, selected_positions] = 0
        
        return attention_mask


def create_efficient_pattern(
    pattern_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create efficient attention patterns
    
    Args:
        pattern_type: Type of pattern ('local_global', 'sliding_window', 
                                      'hierarchical', 'strided')
        **kwargs: Additional arguments for pattern
    
    Returns:
        Efficient pattern instance
    """
    if pattern_type == 'local_global':
        return LocalGlobalPattern(**kwargs)
    elif pattern_type == 'sliding_window':
        return SlidingWindowPattern(**kwargs)
    elif pattern_type == 'hierarchical':
        return HierarchicalPattern(**kwargs)
    elif pattern_type == 'strided':
        return StridedPattern(**kwargs)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")


def analyze_pattern_coverage(
    pattern: nn.Module, 
    seq_len: int, 
    batch_size: int = 1,
    d_model: int = 256
) -> dict:
    """
    Analyze coverage and efficiency of attention pattern
    
    Args:
        pattern: Attention pattern to analyze
        seq_len: Sequence length
        batch_size: Batch size for analysis
        d_model: Model dimension
    
    Returns:
        Dictionary with coverage statistics
    """
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Get attention mask
    with torch.no_grad():
        mask = pattern(x)
    
    # Analyze coverage
    total_positions = seq_len * seq_len
    attended_positions = (mask != float('-inf')).sum().item()
    coverage_ratio = attended_positions / total_positions
    
    # Analyze per-query statistics
    per_query_coverage = []
    for q in range(seq_len):
        query_coverage = (mask[0, 0, q, :] != float('-inf')).sum().item()
        per_query_coverage.append(query_coverage)
    
    avg_per_query = sum(per_query_coverage) / len(per_query_coverage)
    
    return {
        'total_coverage': coverage_ratio,
        'avg_per_query': avg_per_query,
        'min_per_query': min(per_query_coverage),
        'max_per_query': max(per_query_coverage),
        'total_attended': attended_positions,
        'total_possible': total_positions
    }
