"""
Advanced Optimization Strategies for Comprehensive Ablations

This module implements additional advanced optimization strategies that weren't
included in the basic experiment but are crucial for comprehensive ablations.

Advanced Strategies:
1. Learned Attention Patterns
2. Multi-Scale Indexers
3. Dynamic Architecture Selection
4. Hardware-Aware Optimizations
5. Memory-Efficient Variants
6. Gradient-Based Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import math


class LearnedAttentionPattern(nn.Module):
    """
    Learned attention pattern that adapts during training
    
    Instead of fixed patterns, this learns optimal attention patterns
    end-to-end through differentiable operations.
    """
    def __init__(
        self,
        d_model: int,
        pattern_heads: int = 4,
        pattern_dim: int = 32,
        max_patterns: int = 8,
        temperature: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.pattern_heads = pattern_heads
        self.pattern_dim = pattern_dim
        self.max_patterns = max_patterns
        self.temperature = temperature
        
        # Pattern generators
        self.pattern_generators = nn.ModuleList([
            nn.Linear(d_model, pattern_dim) for _ in range(pattern_heads)
        ])
        
        # Pattern selectors
        self.pattern_selector = nn.Linear(d_model, max_patterns)
        
        # Learnable pattern templates
        self.pattern_templates = nn.Parameter(
            torch.randn(max_patterns, pattern_dim, pattern_dim)
        )
        
        # Pattern combination weights
        self.combination_weights = nn.Linear(d_model, pattern_heads)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate learned attention patterns
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            attention_pattern: Learned attention mask [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Generate pattern features for each head
        pattern_features = []
        for head_idx in range(self.pattern_heads):
            features = self.pattern_generators[head_idx](x)  # [batch, seq_len, pattern_dim]
            pattern_features.append(features)
        
        # Select pattern templates
        pattern_logits = self.pattern_selector(x)  # [batch, seq_len, max_patterns]
        pattern_probs = F.softmax(pattern_logits / self.temperature, dim=-1)
        
        # Generate attention patterns
        attention_patterns = []
        for head_idx in range(self.pattern_heads):
            head_pattern = torch.zeros(batch_size, seq_len, seq_len, device=device)
            
            for b in range(batch_size):
                for q in range(seq_len):
                    # Select pattern template for this query
                    selected_patterns = pattern_probs[b, q]  # [max_patterns]
                    
                    # Combine pattern templates
                    combined_pattern = torch.zeros(self.pattern_dim, self.pattern_dim, device=device)
                    for p_idx in range(self.max_patterns):
                        combined_pattern += selected_patterns[p_idx] * self.pattern_templates[p_idx]
                    
                    # Generate attention weights
                    query_features = pattern_features[head_idx][b, q]  # [pattern_dim]
                    key_features = pattern_features[head_idx][b, :q+1]  # [seq_len, pattern_dim] (causal)
                    
                    # Compute attention scores
                    attention_scores = torch.einsum('d,pd->p', query_features, combined_pattern)
                    attention_scores = torch.einsum('p,sp->s', attention_scores, key_features)
                    
                    # Apply causal masking and softmax
                    attention_weights = F.softmax(attention_scores, dim=-1)
                    head_pattern[b, q, :q+1] = attention_weights
            
            attention_patterns.append(head_pattern)
        
        # Combine patterns from different heads
        combination_weights = self.combination_weights(x)  # [batch, seq_len, pattern_heads]
        final_pattern = torch.zeros(batch_size, seq_len, seq_len, device=device)
        
        for head_idx in range(self.pattern_heads):
            head_weight = combination_weights[:, :, head_idx:head_idx+1]  # [batch, seq_len, 1]
            final_pattern += head_weight * attention_patterns[head_idx]
        
        # Convert to attention mask format
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device)
        attention_mask[:, 0, :, :] = final_pattern
        
        return attention_mask


class MultiScaleIndexer(nn.Module):
    """
    Multi-scale indexer that operates at different scales simultaneously
    
    Uses different indexer configurations for different sequence scales
    to optimize both local and global attention patterns.
    """
    def __init__(
        self,
        d_model: int,
        scales: List[int] = [32, 64, 128],
        indexer_heads: List[int] = [2, 4, 8],
        indexer_dims: List[int] = [16, 32, 64]
    ):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        self.num_scales = len(scales)
        
        # Create indexers for each scale
        self.scale_indexers = nn.ModuleList()
        for i in range(self.num_scales):
            indexer = nn.ModuleDict({
                'q_proj': nn.Linear(d_model, indexer_heads[i] * indexer_dims[i], bias=False),
                'k_proj': nn.Linear(d_model, indexer_dims[i], bias=False),
                'w_proj': nn.Linear(d_model, indexer_heads[i], bias=False),
            })
            self.scale_indexers.append(indexer)
        
        # Scale combination weights
        self.scale_combiner = nn.Linear(d_model, self.num_scales)
        
        # Scale-specific parameters
        self.scale_heads = indexer_heads
        self.scale_dims = indexer_dims
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale index scores
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            index_scores: Combined multi-scale index scores [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Compute scale combination weights
        scale_weights = F.softmax(self.scale_combiner(x), dim=-1)  # [batch, seq_len, num_scales]
        
        # Compute index scores for each scale
        scale_scores = []
        for scale_idx in range(self.num_scales):
            scale = self.scales[scale_idx]
            indexer = self.scale_indexers[scale_idx]
            heads = self.scale_heads[scale_idx]
            dim = self.scale_dims[scale_idx]
            
            # Compute queries, keys, weights
            queries = indexer['q_proj'](x).reshape(batch_size, seq_len, heads, dim)
            keys = indexer['k_proj'](x)
            weights = indexer['w_proj'](x)
            
            # Compute dot products
            dots = torch.einsum('bthd,bsd->bths', queries, keys)
            activated = F.relu(dots)
            weighted = activated * weights.unsqueeze(-1)
            scale_score = weighted.sum(dim=2)  # [batch, seq_len, seq_len]
            
            # Apply scale-specific masking
            if scale < seq_len:
                # For smaller scales, focus on local attention
                scale_mask = torch.zeros_like(scale_score)
                for q in range(seq_len):
                    start = max(0, q - scale + 1)
                    end = min(seq_len, q + 1)
                    scale_mask[:, q, start:end] = 1.0
                scale_score = scale_score * scale_mask
            
            scale_scores.append(scale_score)
        
        # Combine scale scores
        combined_scores = torch.zeros_like(scale_scores[0])
        for scale_idx, scale_score in enumerate(scale_scores):
            scale_weight = scale_weights[:, :, scale_idx:scale_idx+1]  # [batch, seq_len, 1]
            combined_scores += scale_weight * scale_score
        
        return combined_scores


class DynamicArchitectureSelector(nn.Module):
    """
    Dynamic architecture selector that chooses optimal configuration based on input
    
    Learns to select different indexer configurations based on sequence characteristics
    like length, complexity, and content.
    """
    def __init__(
        self,
        d_model: int,
        configurations: List[Dict[str, Any]],
        selector_dim: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.configurations = configurations
        self.num_configs = len(configurations)
        
        # Architecture selector
        self.selector = nn.Sequential(
            nn.Linear(d_model, selector_dim),
            nn.ReLU(),
            nn.Linear(selector_dim, selector_dim),
            nn.ReLU(),
            nn.Linear(selector_dim, self.num_configs)
        )
        
        # Create indexers for each configuration
        self.config_indexers = nn.ModuleList()
        for config in configurations:
            indexer = nn.ModuleDict({
                'q_proj': nn.Linear(d_model, config['heads'] * config['dim'], bias=False),
                'k_proj': nn.Linear(d_model, config['dim'], bias=False),
                'w_proj': nn.Linear(d_model, config['heads'], bias=False),
            })
            self.config_indexers.append(indexer)
        
        # Configuration metadata
        self.config_heads = [config['heads'] for config in configurations]
        self.config_dims = [config['dim'] for config in configurations]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Select and apply optimal indexer configuration
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            index_scores: Dynamically selected index scores [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Compute sequence characteristics
        seq_features = self._compute_sequence_features(x)  # [batch, seq_len, d_model]
        
        # Select configuration
        config_logits = self.selector(seq_features)  # [batch, seq_len, num_configs]
        config_probs = F.softmax(config_logits, dim=-1)
        
        # Compute index scores for each configuration
        config_scores = []
        for config_idx in range(self.num_configs):
            indexer = self.config_indexers[config_idx]
            heads = self.config_heads[config_idx]
            dim = self.config_dims[config_idx]
            
            # Compute queries, keys, weights
            queries = indexer['q_proj'](x).reshape(batch_size, seq_len, heads, dim)
            keys = indexer['k_proj'](x)
            weights = indexer['w_proj'](x)
            
            # Compute index scores
            dots = torch.einsum('bthd,bsd->bths', queries, keys)
            activated = F.relu(dots)
            weighted = activated * weights.unsqueeze(-1)
            config_score = weighted.sum(dim=2)
            
            config_scores.append(config_score)
        
        # Combine configurations based on selection probabilities
        final_scores = torch.zeros_like(config_scores[0])
        for config_idx, config_score in enumerate(config_scores):
            config_prob = config_probs[:, :, config_idx:config_idx+1]  # [batch, seq_len, 1]
            final_scores += config_prob * config_score
        
        return final_scores
    
    def _compute_sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sequence-level features for configuration selection"""
        # Simple features: mean, std, and position
        seq_mean = x.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        seq_std = x.std(dim=1, keepdim=True)    # [batch, 1, d_model]
        
        # Position features
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).float() / seq_len
        position_features = position_ids.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, d_model)
        
        # Combine features
        combined_features = x + seq_mean + seq_std + position_features * 0.1
        
        return combined_features


class MemoryEfficientIndexer(nn.Module):
    """
    Memory-efficient indexer that reduces memory usage through various techniques
    
    Implements gradient checkpointing, activation offloading, and other memory optimizations.
    """
    def __init__(
        self,
        d_model: int,
        indexer_heads: int = 2,
        indexer_dim: int = 32,
        use_checkpointing: bool = True,
        use_activation_offload: bool = False,
        chunk_size: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        self.use_checkpointing = use_checkpointing
        self.use_activation_offload = use_activation_offload
        self.chunk_size = chunk_size
        
        # Standard indexer components
        self.q_proj = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.w_proj = nn.Linear(d_model, indexer_heads, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            index_scores: Index scores [batch_size, seq_len, seq_len]
        """
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of forward pass"""
        batch_size, seq_len, _ = x.shape
        
        if self.use_activation_offload and seq_len > self.chunk_size:
            return self._forward_chunked(x)
        
        # Standard forward pass
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        keys = self.k_proj(x)
        weights = self.w_proj(x)
        
        dots = torch.einsum('bthd,bsd->bths', queries, keys)
        activated = F.relu(dots)
        weighted = activated * weights.unsqueeze(-1)
        index_scores = weighted.sum(dim=2)
        
        return index_scores
    
    def _forward_chunked(self, x: torch.Tensor) -> torch.Tensor:
        """Chunked forward pass for memory efficiency"""
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Process in chunks to reduce memory usage
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        chunk_results = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, seq_len)
            
            # Process chunk
            x_chunk = x[:, start_idx:end_idx, :]
            
            queries = self.q_proj(x_chunk).reshape(batch_size, end_idx - start_idx, self.indexer_heads, self.indexer_dim)
            keys = self.k_proj(x)
            weights = self.w_proj(x_chunk)
            
            # Compute scores for this chunk
            dots = torch.einsum('bthd,bsd->bths', queries, keys)
            activated = F.relu(dots)
            weighted = activated * weights.unsqueeze(-1)
            chunk_score = weighted.sum(dim=2)
            
            chunk_results.append(chunk_score)
        
        # Combine chunk results
        index_scores = torch.cat(chunk_results, dim=1)
        
        return index_scores


class GradientBasedOptimizer(nn.Module):
    """
    Gradient-based indexer optimizer that adapts based on training gradients
    
    Uses gradient information to dynamically adjust indexer behavior during training.
    """
    def __init__(
        self,
        d_model: int,
        indexer_heads: int = 2,
        indexer_dim: int = 32,
        adaptation_rate: float = 0.01,
        gradient_window: int = 100
    ):
        super().__init__()
        self.d_model = d_model
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        self.adaptation_rate = adaptation_rate
        self.gradient_window = gradient_window
        
        # Standard indexer components
        self.q_proj = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
        self.w_proj = nn.Linear(d_model, indexer_heads, bias=False)
        
        # Adaptation parameters
        self.adaptation_weights = nn.Parameter(torch.ones(indexer_heads))
        self.gradient_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradient-based adaptation
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            index_scores: Adapted index scores [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute index scores
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        keys = self.k_proj(x)
        weights = self.w_proj(x)
        
        dots = torch.einsum('bthd,bsd->bths', queries, keys)
        activated = F.relu(dots)
        weighted = activated * weights.unsqueeze(-1)
        
        # Apply adaptation weights
        adapted_weighted = weighted * self.adaptation_weights.view(1, 1, -1, 1)
        
        index_scores = adapted_weighted.sum(dim=2)
        
        return index_scores
    
    def update_adaptation(self, gradients: torch.Tensor):
        """Update adaptation weights based on gradients"""
        if len(self.gradient_history) >= self.gradient_window:
            self.gradient_history.pop(0)
        
        self.gradient_history.append(gradients.detach())
        
        if len(self.gradient_history) > 10:  # Need some history
            # Compute adaptation direction
            recent_gradients = torch.stack(self.gradient_history[-10:])
            adaptation_direction = recent_gradients.mean(dim=0)
            
            # Update adaptation weights
            with torch.no_grad():
                self.adaptation_weights += self.adaptation_rate * adaptation_direction
                self.adaptation_weights.clamp_(0.1, 10.0)  # Prevent extreme values


def create_advanced_indexer(
    indexer_type: str,
    d_model: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create advanced indexer variants
    
    Args:
        indexer_type: Type of advanced indexer
        d_model: Model dimension
        **kwargs: Additional arguments
    
    Returns:
        Advanced indexer instance
    """
    if indexer_type == 'learned_pattern':
        return LearnedAttentionPattern(d_model, **kwargs)
    elif indexer_type == 'multiscale':
        return MultiScaleIndexer(d_model, **kwargs)
    elif indexer_type == 'dynamic_arch':
        return DynamicArchitectureSelector(d_model, **kwargs)
    elif indexer_type == 'memory_efficient':
        return MemoryEfficientIndexer(d_model, **kwargs)
    elif indexer_type == 'gradient_based':
        return GradientBasedOptimizer(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown advanced indexer type: {indexer_type}")


def benchmark_advanced_indexers(
    indexers: Dict[str, nn.Module],
    x: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark advanced indexer variants
    
    Args:
        indexers: Dictionary of indexer name -> indexer instance
        x: Input tensor
        num_runs: Number of benchmark runs
    
    Returns:
        Benchmark results for each indexer
    """
    results = {}
    
    for name, indexer in indexers.items():
        print(f"Benchmarking {name}...")
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = indexer(x)
        
        # Benchmark
        device = next(indexer.parameters()).device
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        
        if start_time is not None:
            start_time.record()
        else:
            import time
            start_time_val = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = indexer(x)
        
        if end_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / num_runs
        else:
            elapsed_time = (time.time() - start_time_val) * 1000 / num_runs
        
        # Memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
            memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        else:
            memory_allocated = memory_reserved = 0
        
        # Parameter count
        total_params = sum(p.numel() for p in indexer.parameters())
        
        results[name] = {
            'avg_time_ms': elapsed_time,
            'total_params': total_params,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'throughput_tokens_per_sec': x.numel() / (elapsed_time / 1000)
        }
    
    return results
