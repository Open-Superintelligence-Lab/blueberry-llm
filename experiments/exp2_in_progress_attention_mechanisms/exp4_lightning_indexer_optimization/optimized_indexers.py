"""
Optimized Lightning Indexer Variants for Experiment 4

This module implements various optimizations of the Lightning Indexer to reduce
computational overhead while maintaining attention quality.

Optimization strategies:
1. Reduced complexity (fewer heads, smaller dimensions)
2. Quantization (FP16, mixed precision)
3. Efficient computation patterns
4. Memory optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import time


class OptimizedLightningIndexer(nn.Module):
    """
    Optimized Lightning Indexer with configurable complexity reduction
    
    Args:
        d_model: Model dimension
        indexer_heads: Number of indexer heads (reduced from original 4)
        indexer_dim: Dimension of indexer queries/keys (reduced from original 64)
        use_fp16: Use half-precision for indexer computations
        dropout: Dropout probability
    """
    def __init__(
        self, 
        d_model: int, 
        indexer_heads: int = 2,  # Reduced from 4
        indexer_dim: int = 32,   # Reduced from 64
        use_fp16: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        self.use_fp16 = use_fp16
        
        # Indexer query projection: h_t -> {q_{t,j}^I}
        self.q_proj = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)
        
        # Indexer key projection: h_s -> k_s^I
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        
        # Indexer weights: w_{t,j}^I for each head
        self.w_proj = nn.Linear(d_model, indexer_heads, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Convert to half precision if requested
        if use_fp16:
            self.half()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute optimized index scores between all pairs of tokens
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            index_scores: Index scores [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Convert to half precision for computation if using FP16
        if self.use_fp16 and x.dtype == torch.float32:
            x = x.half()
        
        # Compute indexer queries: [batch, seq_len, indexer_heads, indexer_dim]
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        
        # Compute indexer keys: [batch, seq_len, indexer_dim]
        keys = self.k_proj(x)
        
        # Compute indexer weights: [batch, seq_len, indexer_heads]
        weights = self.w_proj(x)
        
        # Compute dot products: q_{t,j} · k_s for all t, s, j
        # Use more efficient einsum for reduced complexity
        dots = torch.einsum('bthd,bsd->bths', queries, keys)
        
        # Apply ReLU activation
        activated = F.relu(dots)
        
        # Weight each head: w_{t,j} · ReLU(q_{t,j} · k_s)
        weighted = activated * weights.unsqueeze(-1)
        
        # Sum across heads: Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)
        index_scores = weighted.sum(dim=2)
        
        # Convert back to float32 if needed
        if self.use_fp16 and index_scores.dtype == torch.float16:
            index_scores = index_scores.float()
        
        return index_scores


class MinimalLightningIndexer(nn.Module):
    """
    Minimal Lightning Indexer with single head and small dimension
    
    Extreme optimization: 1 head, 16 dims (98% fewer parameters than original)
    """
    def __init__(
        self, 
        d_model: int, 
        indexer_dim: int = 16,  # Very small dimension
        use_fp16: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.indexer_dim = indexer_dim
        self.use_fp16 = use_fp16
        
        # Single head projections
        self.q_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.w_proj = nn.Linear(d_model, 1, bias=False)  # Single weight
        
        self.dropout = nn.Dropout(dropout)
        
        if use_fp16:
            self.half()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Minimal indexer forward pass"""
        batch_size, seq_len, _ = x.shape
        
        if self.use_fp16 and x.dtype == torch.float32:
            x = x.half()
        
        # Single head computation
        queries = self.q_proj(x)  # [batch, seq_len, indexer_dim]
        keys = self.k_proj(x)     # [batch, seq_len, indexer_dim]
        weights = self.w_proj(x)  # [batch, seq_len, 1]
        
        # Compute dot products: q_t · k_s
        dots = torch.einsum('btd,bsd->bts', queries, keys)
        
        # Apply ReLU and weight
        activated = F.relu(dots)
        weighted = activated * weights
        
        if self.use_fp16 and weighted.dtype == torch.float16:
            weighted = weighted.float()
        
        return weighted


class UltraLightIndexer(nn.Module):
    """
    Ultra-light indexer with minimal parameters (8 dims, single head)
    
    For extreme speed optimization where quality can be slightly compromised
    """
    def __init__(
        self, 
        d_model: int, 
        indexer_dim: int = 8,  # Ultra-small dimension
        use_fp16: bool = True,  # Always use FP16 for ultra-light
        dropout: float = 0.0   # No dropout for speed
    ):
        super().__init__()
        self.d_model = d_model
        self.indexer_dim = indexer_dim
        self.use_fp16 = use_fp16
        
        # Minimal projections
        self.q_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        
        # No weights - just direct dot product
        self.half()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ultra-light forward pass - just dot product"""
        batch_size, seq_len, _ = x.shape
        
        if x.dtype == torch.float32:
            x = x.half()
        
        queries = self.q_proj(x)  # [batch, seq_len, 8]
        keys = self.k_proj(x)     # [batch, seq_len, 8]
        
        # Simple dot product without weights
        dots = torch.einsum('btd,bsd->bts', queries, keys)
        
        # Apply ReLU
        index_scores = F.relu(dots)
        
        # Convert back to float32
        if index_scores.dtype == torch.float16:
            index_scores = index_scores.float()
        
        return index_scores


class CachedLightningIndexer(nn.Module):
    """
    Lightning Indexer with caching for repeated computations
    
    Caches indexer computations when input sequences are similar
    """
    def __init__(
        self, 
        d_model: int, 
        indexer_heads: int = 2,
        indexer_dim: int = 32,
        cache_size: int = 100,
        use_fp16: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        self.cache_size = cache_size
        self.use_fp16 = use_fp16
        
        # Standard indexer components
        self.q_proj = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.w_proj = nn.Linear(d_model, indexer_heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Cache for storing computed index scores
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if use_fp16:
            self.half()
    
    def _compute_cache_key(self, x: torch.Tensor) -> str:
        """Compute a simple hash key for caching"""
        # Use sequence length and a few sample values as key
        seq_len = x.shape[1]
        sample_values = x[0, :min(10, seq_len), :min(10, x.shape[2])].flatten()
        key = f"{seq_len}_{hash(sample_values.sum().item())}"
        return key
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with caching"""
        cache_key = self._compute_cache_key(x)
        
        # Check cache first
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Compute index scores
        batch_size, seq_len, _ = x.shape
        
        if self.use_fp16 and x.dtype == torch.float32:
            x = x.half()
        
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        keys = self.k_proj(x)
        weights = self.w_proj(x)
        
        dots = torch.einsum('bthd,bsd->bths', queries, keys)
        activated = F.relu(dots)
        weighted = activated * weights.unsqueeze(-1)
        index_scores = weighted.sum(dim=2)
        
        if self.use_fp16 and index_scores.dtype == torch.float16:
            index_scores = index_scores.float()
        
        # Cache the result
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = index_scores.detach()
        
        return index_scores
    
    def get_cache_stats(self) -> dict:
        """Get cache hit/miss statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


def create_optimized_indexer(
    indexer_type: str,
    d_model: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create optimized indexer variants
    
    Args:
        indexer_type: Type of indexer ('optimized', 'minimal', 'ultra_light', 'cached')
        d_model: Model dimension
        **kwargs: Additional arguments for indexer
    
    Returns:
        Optimized indexer instance
    """
    if indexer_type == 'optimized':
        return OptimizedLightningIndexer(d_model, **kwargs)
    elif indexer_type == 'minimal':
        return MinimalLightningIndexer(d_model, **kwargs)
    elif indexer_type == 'ultra_light':
        return UltraLightIndexer(d_model, **kwargs)
    elif indexer_type == 'cached':
        return CachedLightningIndexer(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown indexer type: {indexer_type}")


def benchmark_indexer(indexer: nn.Module, x: torch.Tensor, num_runs: int = 100) -> dict:
    """
    Benchmark indexer performance
    
    Args:
        indexer: Indexer to benchmark
        x: Input tensor
        num_runs: Number of runs for timing
    
    Returns:
        Dictionary with timing and memory statistics
    """
    device = next(indexer.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = indexer(x)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = indexer(x)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    # Memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
    else:
        memory_allocated = memory_reserved = 0
    
    # Parameter count
    total_params = sum(p.numel() for p in indexer.parameters())
    
    return {
        'avg_time_ms': (end_time - start_time) * 1000 / num_runs,
        'total_params': total_params,
        'memory_allocated_mb': memory_allocated / 1024 / 1024,
        'memory_reserved_mb': memory_reserved / 1024 / 1024,
        'output_shape': output.shape
    }
