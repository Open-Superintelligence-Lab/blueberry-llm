"""
Adaptive Selection Strategies for Experiment 4

This module implements adaptive k-value selection strategies that dynamically
adjust the number of tokens to attend to based on sequence characteristics.

Strategies:
1. Fixed ratio: k = ratio * sequence_length
2. Progressive: k increases during training
3. Entropy-based: k based on sequence complexity/entropy
4. Position-aware: k varies by position in sequence
5. Dynamic: k adapts based on attention scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Callable
import math


class AdaptiveKSelector(nn.Module):
    """
    Adaptive k-value selector that dynamically adjusts k based on sequence characteristics
    
    Args:
        base_k: Base number of tokens to select
        min_k: Minimum k value
        max_k: Maximum k value
        adaptation_strategy: Strategy for adapting k ('fixed', 'progressive', 'entropy', 'position', 'dynamic')
        d_model: Model dimension
    """
    def __init__(
        self,
        base_k: int = 64,
        min_k: int = 16,
        max_k: int = 512,
        adaptation_strategy: str = 'fixed',
        d_model: int = 256,
        indexer_dim: int = 32
    ):
        super().__init__()
        self.base_k = base_k
        self.min_k = min_k
        self.max_k = max_k
        self.adaptation_strategy = adaptation_strategy
        self.d_model = d_model
        
        # Indexer for computing attention scores
        self.q_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        
        # Entropy-based adaptation components
        if adaptation_strategy == 'entropy':
            self.entropy_proj = nn.Linear(d_model, 1, bias=False)
        
        # Position-aware adaptation
        elif adaptation_strategy == 'position':
            self.position_proj = nn.Linear(d_model, 1, bias=False)
        
        # Dynamic adaptation
        elif adaptation_strategy == 'dynamic':
            self.dynamic_proj = nn.Linear(d_model, 1, bias=False)
        
        # Progressive adaptation state
        self.training_step = 0
        self.max_training_steps = 1000
        
    def forward(
        self, 
        x: torch.Tensor, 
        index_scores: torch.Tensor,
        training_step: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select adaptive k tokens based on strategy
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            index_scores: Pre-computed index scores [batch_size, seq_len, seq_len]
            training_step: Current training step (for progressive adaptation)
            
        Returns:
            top_k_mask: Boolean mask for selected tokens [batch_size, seq_len, seq_len]
            k_values: Actual k values used [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Determine k values based on strategy
        k_values = self._compute_k_values(x, seq_len, training_step)
        
        # Create top-k mask
        top_k_mask = torch.zeros(
            batch_size, seq_len, seq_len,
            dtype=torch.bool, device=device
        )
        
        # Apply causal masking to index scores
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        masked_scores = index_scores.masked_fill(~causal_mask, float('-inf'))
        
        # Select top-k tokens for each query
        for b in range(batch_size):
            for q in range(seq_len):
                k = k_values[b, q].item()
                k = min(k, q + 1)  # Can't attend to more tokens than available
                
                if k > 0:
                    _, top_k_indices = torch.topk(masked_scores[b, q], k, dim=-1)
                    top_k_mask[b, q, top_k_indices] = True
        
        return top_k_mask, k_values
    
    def _compute_k_values(
        self, 
        x: torch.Tensor, 
        seq_len: int, 
        training_step: Optional[int] = None
    ) -> torch.Tensor:
        """Compute k values based on adaptation strategy"""
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        if self.adaptation_strategy == 'fixed':
            # Fixed ratio of sequence length
            k_values = torch.full(
                (batch_size, seq_len), 
                min(self.base_k, seq_len),
                dtype=torch.long, device=device
            )
        
        elif self.adaptation_strategy == 'progressive':
            # Progressive k: starts small, increases during training
            if training_step is not None:
                self.training_step = training_step
            
            progress = min(1.0, self.training_step / self.max_training_steps)
            current_k = int(self.min_k + (self.max_k - self.min_k) * progress)
            current_k = min(current_k, seq_len)
            
            k_values = torch.full(
                (batch_size, seq_len), 
                current_k,
                dtype=torch.long, device=device
            )
        
        elif self.adaptation_strategy == 'entropy':
            # k based on sequence entropy/complexity
            k_values = self._entropy_based_k(x, seq_len)
        
        elif self.adaptation_strategy == 'position':
            # k varies by position in sequence
            k_values = self._position_based_k(x, seq_len)
        
        elif self.adaptation_strategy == 'dynamic':
            # k adapts based on attention scores
            k_values = self._dynamic_k(x, seq_len)
        
        else:
            raise ValueError(f"Unknown adaptation strategy: {self.adaptation_strategy}")
        
        # Clamp k values to valid range
        k_values = torch.clamp(k_values, self.min_k, min(self.max_k, seq_len))
        
        return k_values
    
    def _entropy_based_k(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute k based on sequence entropy"""
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Compute sequence entropy
        entropy_scores = self.entropy_proj(x).squeeze(-1)  # [batch, seq_len]
        entropy_scores = torch.sigmoid(entropy_scores)  # Normalize to [0, 1]
        
        # Map entropy to k values
        k_range = self.max_k - self.min_k
        k_values = self.min_k + entropy_scores * k_range
        
        return k_values.long()
    
    def _position_based_k(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute k based on position in sequence"""
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Create position embeddings
        positions = torch.arange(seq_len, device=device).float()
        positions = positions / seq_len  # Normalize to [0, 1]
        
        # Compute position-based k
        k_values = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            for p in range(seq_len):
                # Earlier positions need more context, later positions need less
                position_factor = 1.0 - positions[p] * 0.5  # Reduce k for later positions
                k = int(self.base_k * position_factor)
                k_values[b, p] = k
        
        return k_values.long()
    
    def _dynamic_k(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute k dynamically based on attention scores"""
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Compute attention scores
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        scores = torch.einsum('bqd,bkd->bqk', queries, keys)
        scores = F.relu(scores)
        
        # Apply causal masking
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        masked_scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Compute k based on score distribution
        k_values = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            for q in range(seq_len):
                query_scores = masked_scores[b, q, :q+1]
                if query_scores.numel() > 0:
                    # Use score variance to determine k
                    score_variance = torch.var(query_scores)
                    variance_factor = torch.sigmoid(score_variance)
                    
                    k = int(self.min_k + (self.max_k - self.min_k) * variance_factor)
                    k_values[b, q] = k
        
        return k_values.long()


class FixedRatioSelector(nn.Module):
    """
    Simple fixed ratio selector: k = ratio * sequence_length
    
    Args:
        ratio: Ratio of sequence length to use for k (e.g., 0.5 = 50%)
        min_k: Minimum k value
        max_k: Maximum k value
    """
    def __init__(
        self,
        ratio: float = 0.5,
        min_k: int = 16,
        max_k: int = 512
    ):
        super().__init__()
        self.ratio = ratio
        self.min_k = min_k
        self.max_k = max_k
    
    def forward(
        self, 
        x: torch.Tensor, 
        index_scores: torch.Tensor,
        training_step: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select tokens using fixed ratio"""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Compute k based on ratio
        k = int(seq_len * self.ratio)
        k = max(self.min_k, min(self.max_k, k))
        
        # Create top-k mask
        top_k_mask = torch.zeros(
            batch_size, seq_len, seq_len,
            dtype=torch.bool, device=device
        )
        
        # Apply causal masking
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        masked_scores = index_scores.masked_fill(~causal_mask, float('-inf'))
        
        # Select top-k tokens
        for b in range(batch_size):
            for q in range(seq_len):
                query_k = min(k, q + 1)
                if query_k > 0:
                    _, top_k_indices = torch.topk(masked_scores[b, q], query_k, dim=-1)
                    top_k_mask[b, q, top_k_indices] = True
        
        k_values = torch.full(
            (batch_size, seq_len), k, dtype=torch.long, device=device
        )
        
        return top_k_mask, k_values


class ProgressiveSelector(nn.Module):
    """
    Progressive selector that increases k during training
    
    Args:
        start_k: Starting k value
        end_k: Final k value
        max_steps: Maximum training steps for progression
        progression_type: Type of progression ('linear', 'exponential', 'cosine')
    """
    def __init__(
        self,
        start_k: int = 16,
        end_k: int = 256,
        max_steps: int = 1000,
        progression_type: str = 'linear'
    ):
        super().__init__()
        self.start_k = start_k
        self.end_k = end_k
        self.max_steps = max_steps
        self.progression_type = progression_type
        self.training_step = 0
    
    def forward(
        self, 
        x: torch.Tensor, 
        index_scores: torch.Tensor,
        training_step: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select tokens with progressive k"""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Update training step
        if training_step is not None:
            self.training_step = training_step
        
        # Compute current k based on progression
        progress = min(1.0, self.training_step / self.max_steps)
        
        if self.progression_type == 'linear':
            current_k = self.start_k + (self.end_k - self.start_k) * progress
        elif self.progression_type == 'exponential':
            current_k = self.start_k * ((self.end_k / self.start_k) ** progress)
        elif self.progression_type == 'cosine':
            current_k = self.start_k + (self.end_k - self.start_k) * (1 - math.cos(progress * math.pi)) / 2
        else:
            raise ValueError(f"Unknown progression type: {self.progression_type}")
        
        current_k = int(current_k)
        current_k = min(current_k, seq_len)
        
        # Create top-k mask
        top_k_mask = torch.zeros(
            batch_size, seq_len, seq_len,
            dtype=torch.bool, device=device
        )
        
        # Apply causal masking
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        masked_scores = index_scores.masked_fill(~causal_mask, float('-inf'))
        
        # Select top-k tokens
        for b in range(batch_size):
            for q in range(seq_len):
                query_k = min(current_k, q + 1)
                if query_k > 0:
                    _, top_k_indices = torch.topk(masked_scores[b, q], query_k, dim=-1)
                    top_k_mask[b, q, top_k_indices] = True
        
        k_values = torch.full(
            (batch_size, seq_len), current_k, dtype=torch.long, device=device
        )
        
        return top_k_mask, k_values


def create_adaptive_selector(
    selector_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create adaptive selectors
    
    Args:
        selector_type: Type of selector ('adaptive', 'fixed_ratio', 'progressive')
        **kwargs: Additional arguments for selector
    
    Returns:
        Adaptive selector instance
    """
    if selector_type == 'adaptive':
        return AdaptiveKSelector(**kwargs)
    elif selector_type == 'fixed_ratio':
        return FixedRatioSelector(**kwargs)
    elif selector_type == 'progressive':
        return ProgressiveSelector(**kwargs)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")


def analyze_k_distribution(
    selector: nn.Module,
    x: torch.Tensor,
    index_scores: torch.Tensor,
    training_step: Optional[int] = None
) -> Dict[str, float]:
    """
    Analyze the distribution of k values produced by selector
    
    Args:
        selector: Selector to analyze
        x: Input tensor
        index_scores: Pre-computed index scores
        training_step: Training step for progressive selectors
    
    Returns:
        Dictionary with k distribution statistics
    """
    with torch.no_grad():
        _, k_values = selector(x, index_scores, training_step)
    
    k_values_flat = k_values.flatten()
    
    return {
        'mean_k': k_values_flat.float().mean().item(),
        'std_k': k_values_flat.float().std().item(),
        'min_k': k_values_flat.min().item(),
        'max_k': k_values_flat.max().item(),
        'median_k': k_values_flat.float().median().item()
    }
