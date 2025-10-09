"""
Model Definitions for Experiment 4

This module contains model definitions for the Lightning Indexer optimization experiment.
It provides a unified interface for creating models with different optimization strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import sys
import os

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

from optimized_indexers import create_optimized_indexer
from efficient_patterns import create_efficient_pattern
from adaptive_selection import create_adaptive_selector


class OptimizedSparseAttentionLayer(nn.Module):
    """
    Single transformer layer with optimized sparse attention
    
    This layer can use different optimization strategies:
    - Reduced complexity indexers
    - Efficient attention patterns
    - Adaptive k-value selection
    - Quantization techniques
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        optimization_strategy: str = 'baseline',
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.optimization_strategy = optimization_strategy
        
        if strategy_kwargs is None:
            strategy_kwargs = {}
        
        # Standard attention components
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary positional embeddings
        from torchtune.modules import RotaryPositionalEmbeddings
        self.rotary = RotaryPositionalEmbeddings(
            dim=d_model // n_heads,
            max_seq_len=max_seq_len
        )
        
        # Create optimized components based on strategy
        self.indexer = self._create_indexer(strategy_kwargs)
        self.selector = self._create_selector(strategy_kwargs)
        self.efficient_pattern = self._create_pattern(strategy_kwargs)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.use_sparse = True
        
    def _create_indexer(self, kwargs):
        """Create indexer based on optimization strategy"""
        if self.optimization_strategy == 'baseline':
            # Original Lightning Indexer (4 heads, 64 dims)
            return create_optimized_indexer(
                'optimized',
                d_model=self.d_model,
                indexer_heads=4,
                indexer_dim=64,
                use_fp16=False
            )
        elif self.optimization_strategy == 'optimized':
            # Optimized indexer (2 heads, 32 dims)
            return create_optimized_indexer(
                'optimized',
                d_model=self.d_model,
                indexer_heads=2,
                indexer_dim=32,
                use_fp16=kwargs.get('use_fp16', False)
            )
        elif self.optimization_strategy == 'minimal':
            # Minimal indexer (1 head, 16 dims)
            return create_optimized_indexer(
                'minimal',
                d_model=self.d_model,
                indexer_dim=16,
                use_fp16=kwargs.get('use_fp16', False)
            )
        elif self.optimization_strategy == 'ultra_light':
            # Ultra-light indexer (1 head, 8 dims)
            return create_optimized_indexer(
                'ultra_light',
                d_model=self.d_model,
                indexer_dim=8,
                use_fp16=True
            )
        elif self.optimization_strategy == 'fp16_indexer':
            # FP16 indexer (original config with FP16)
            return create_optimized_indexer(
                'optimized',
                d_model=self.d_model,
                indexer_heads=4,
                indexer_dim=64,
                use_fp16=True
            )
        else:
            # Default to optimized
            return create_optimized_indexer(
                'optimized',
                d_model=self.d_model,
                indexer_heads=2,
                indexer_dim=32
            )
    
    def _create_selector(self, kwargs):
        """Create selector based on optimization strategy"""
        if 'fixed_ratio' in self.optimization_strategy:
            ratio = 0.25 if '25' in self.optimization_strategy else 0.5
            return create_adaptive_selector('fixed_ratio', ratio=ratio)
        elif self.optimization_strategy == 'progressive':
            return create_adaptive_selector(
                'progressive',
                start_k=kwargs.get('start_k', 16),
                end_k=kwargs.get('end_k', 256),
                max_steps=kwargs.get('max_steps', 1000)
            )
        else:
            # Default selector (from original sparse attention)
            from experiments.exp1_sparse_vs_classic_attention.sparse_attention import TopKTokenSelector
            return TopKTokenSelector(top_k=kwargs.get('top_k', 64))
    
    def _create_pattern(self, kwargs):
        """Create efficient pattern if needed"""
        if self.optimization_strategy == 'local_global':
            return create_efficient_pattern(
                'local_global',
                local_window=kwargs.get('local_window', 32),
                global_k=kwargs.get('global_k', 64),
                d_model=self.d_model
            )
        elif self.optimization_strategy == 'sliding_window':
            return create_efficient_pattern(
                'sliding_window',
                window_size=kwargs.get('window_size', 64),
                stride=kwargs.get('stride', 32),
                d_model=self.d_model
            )
        elif self.optimization_strategy == 'hierarchical':
            return create_efficient_pattern(
                'hierarchical',
                local_window=kwargs.get('local_window', 16),
                medium_window=kwargs.get('medium_window', 64),
                global_k=kwargs.get('global_k', 32),
                d_model=self.d_model
            )
        else:
            return None
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_index_scores: bool = False
    ):
        """Forward pass with optimized sparse attention"""
        batch_size, seq_len, _ = x.shape
        
        # Pre-norm
        norm_x = self.ln1(x)
        
        # Standard QKV computation
        QKV = self.qkv(norm_x)
        Q, K, V = QKV.split(self.d_model, dim=-1)
        
        # Reshape for multi-head attention
        head_dim = self.d_model // self.n_heads
        Q = Q.reshape(batch_size, seq_len, self.n_heads, head_dim)
        K = K.reshape(batch_size, seq_len, self.n_heads, head_dim)
        V = V.reshape(batch_size, seq_len, self.n_heads, head_dim)
        
        # Apply rotary positional embeddings
        Q = self.rotary(Q)
        K = self.rotary(K)
        
        # Compute index scores
        index_scores = self.indexer(norm_x)
        
        if self.use_sparse:
            if self.efficient_pattern:
                # Use efficient pattern instead of selector
                attention_mask = self.efficient_pattern(norm_x)
                
                # Apply attention with pattern mask
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                    attn_mask=attention_mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
            else:
                # Use selector-based sparse attention
                # Check if selector is adaptive (takes x) or original (takes only index_scores)
                if hasattr(self.selector, 'adaptation_strategy') or hasattr(self.selector, 'ratio'):
                    # Adaptive selector
                    top_k_mask, k_values = self.selector(norm_x, index_scores, training_step=None)
                else:
                    # Original selector
                    top_k_mask, top_k_indices = self.selector(index_scores, apply_causal_mask=True)
                    k_values = torch.full((batch_size, seq_len), self.selector.top_k, dtype=torch.long, device=x.device)
                
                # Create attention mask from top-k selection
                attn_mask = torch.zeros(
                    batch_size, 1, seq_len, seq_len,
                    device=x.device, dtype=Q.dtype
                )
                attn_mask = attn_mask.masked_fill(~top_k_mask.unsqueeze(1), float('-inf'))
                
                # Apply sparse attention
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
        else:
            # Dense attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.w_o(attn_output)
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # FFN with residual connection
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        if return_index_scores:
            return x, index_scores
        return x, None
    
    def enable_sparse(self):
        """Enable sparse attention"""
        self.use_sparse = True
    
    def disable_sparse(self):
        """Disable sparse attention (use dense)"""
        self.use_sparse = False


class OptimizedMoELLM(nn.Module):
    """
    Mixture of Experts LLM with optimized sparse attention
    
    This model can use different optimization strategies for the Lightning Indexer
    while maintaining the same overall architecture.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        optimization_strategy: str = 'baseline',
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.optimization_strategy = optimization_strategy
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers with optimized sparse attention
        self.layers = nn.ModuleList([
            OptimizedSparseAttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                optimization_strategy=optimization_strategy,
                strategy_kwargs=strategy_kwargs,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        return_index_scores: bool = False
    ):
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Store index scores from each layer
        all_index_scores = []
        
        # Transformer layers
        for layer in self.layers:
            hidden_states, index_scores = layer(hidden_states, return_index_scores=return_index_scores)
            if return_index_scores and index_scores is not None:
                all_index_scores.append(index_scores)
        
        # Output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if return_index_scores:
            return logits, all_index_scores
        return logits
    
    def get_parameter_counts(self) -> Dict[str, int]:
        """Get parameter counts for different components"""
        counts = {
            'total': sum(p.numel() for p in self.parameters()),
            'embeddings': sum(p.numel() for p in self.token_embedding.parameters()) + sum(p.numel() for p in self.position_embedding.parameters()),
            'attention': 0,
            'ffn': 0,
            'indexer': 0,
            'output': sum(p.numel() for p in self.ln_f.parameters()) + sum(p.numel() for p in self.lm_head.parameters())
        }
        
        # Count parameters in each layer
        for layer in self.layers:
            counts['attention'] += sum(p.numel() for p in layer.qkv.parameters()) + sum(p.numel() for p in layer.w_o.parameters()) + sum(p.numel() for p in layer.rotary.parameters()) + sum(p.numel() for p in layer.ln1.parameters())
            counts['ffn'] += sum(p.numel() for p in layer.ffn.parameters()) + sum(p.numel() for p in layer.ln2.parameters())
            counts['indexer'] += sum(p.numel() for p in layer.indexer.parameters())
        
        return counts
    
    def enable_sparse_all_layers(self):
        """Enable sparse attention for all layers"""
        for layer in self.layers:
            layer.enable_sparse()
    
    def disable_sparse_all_layers(self):
        """Disable sparse attention for all layers (use dense)"""
        for layer in self.layers:
            layer.disable_sparse()


def create_optimized_model(
    optimization_strategy: str,
    vocab_size: int = 10000,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 512,
    max_seq_len: int = 1024,
    strategy_kwargs: Optional[Dict[str, Any]] = None
) -> OptimizedMoELLM:
    """
    Factory function to create optimized models
    
    Args:
        optimization_strategy: Strategy to use ('baseline', 'optimized', 'minimal', etc.)
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: FFN dimension
        max_seq_len: Maximum sequence length
        strategy_kwargs: Additional arguments for the strategy
    
    Returns:
        Optimized model instance
    """
    return OptimizedMoELLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        optimization_strategy=optimization_strategy,
        strategy_kwargs=strategy_kwargs
    )


def compare_model_sizes(
    strategies: list,
    vocab_size: int = 10000,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 512,
    max_seq_len: int = 1024
) -> Dict[str, Dict[str, int]]:
    """
    Compare model sizes across different optimization strategies
    
    Args:
        strategies: List of strategies to compare
        **model_kwargs: Model configuration parameters
    
    Returns:
        Dictionary mapping strategy names to parameter counts
    """
    results = {}
    
    for strategy in strategies:
        model = create_optimized_model(
            optimization_strategy=strategy,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len
        )
        
        results[strategy] = model.get_parameter_counts()
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


def benchmark_model_forward(
    model: OptimizedMoELLM,
    input_ids: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark model forward pass performance
    
    Args:
        model: Model to benchmark
        input_ids: Input tensor
        num_runs: Number of runs for timing
    
    Returns:
        Dictionary with timing statistics
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)
    
    # Benchmark
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
            logits = model(input_ids)
    
    if end_time is not None:
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / num_runs  # ms
    else:
        elapsed_time = (time.time() - start_time_val) * 1000 / num_runs  # ms
    
    # Memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024    # MB
    else:
        memory_allocated = memory_reserved = 0
    
    return {
        'avg_time_ms': elapsed_time,
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved,
        'throughput_tokens_per_sec': input_ids.numel() / (elapsed_time / 1000)
    }
