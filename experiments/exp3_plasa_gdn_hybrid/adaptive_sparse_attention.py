"""
Adaptive Per-Layer Sparse Attention Implementation

This module implements sparse attention with layer-specific top-k values.
Based on research showing different layers specialize in different functions:
- Early layers: Local patterns, short-range dependencies
- Middle layers: Feature composition, functionally redundant
- Late layers: Global context consolidation, semantic abstraction

Key Innovation: Each layer has a different sparsity budget (k value) optimized
for its functional role in the transformer hierarchy.

References:
- "Learning to Skip the Middle Layers of Transformers" (2025)
- "Transformer Layers as Painters" - Emergence.ai (2025)
- DeepSeek-V3.2-Exp Lightning Indexer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class SparsitySchedule(Enum):
    """Predefined sparsity schedules for different hypotheses"""
    DENSE_BASELINE = "dense_baseline"
    UNIFORM_SPARSE = "uniform_sparse"
    DENSE_TO_SPARSE = "dense_to_sparse"
    AGGRESSIVE_MIDDLE = "aggressive_middle"
    PROGRESSIVE_SPARSE = "progressive_sparse"
    REVERSE_PROGRESSIVE = "reverse_progressive"


@dataclass
class LayerSparsityConfig:
    """Configuration for per-layer sparsity"""
    schedule_name: str
    layer_k_values: List[int]  # k value for each layer
    layer_k_ratios: List[float]  # k as fraction of sequence length
    description: str

    def get_k_for_layer(self, layer_idx: int, seq_len: int) -> int:
        """Get k value for a specific layer"""
        if layer_idx >= len(self.layer_k_ratios):
            # Default to last value if layer index exceeds config
            ratio = self.layer_k_ratios[-1]
        else:
            ratio = self.layer_k_ratios[layer_idx]

        k = int(seq_len * ratio)
        return max(1, min(k, seq_len))  # Clamp to [1, seq_len]


def create_sparsity_schedule(
    schedule: SparsitySchedule,
    n_layers: int,
    seq_len: int
) -> LayerSparsityConfig:
    """
    Create a sparsity schedule based on predefined patterns

    Args:
        schedule: Schedule type
        n_layers: Number of transformer layers
        seq_len: Sequence length

    Returns:
        LayerSparsityConfig with per-layer k values
    """
    if schedule == SparsitySchedule.DENSE_BASELINE:
        # All layers dense (no sparsity)
        ratios = [1.0] * n_layers
        description = "Baseline: All layers dense (k=L)"

    elif schedule == SparsitySchedule.UNIFORM_SPARSE:
        # All layers uniform 50% sparsity (Exp2 baseline)
        ratios = [0.5] * n_layers
        description = "Uniform: All layers k=L/2 (Exp2 baseline)"

    elif schedule == SparsitySchedule.DENSE_TO_SPARSE:
        # Conservative: Dense early, gradually sparse
        # Early (0-33%): Dense (k=L)
        # Middle (33-66%): Moderate sparse (k=L/2)
        # Late (66-100%): Light sparse (k=3L/4)
        ratios = []
        early_cutoff = n_layers // 3
        middle_cutoff = 2 * n_layers // 3

        for i in range(n_layers):
            if i < early_cutoff:
                ratios.append(1.0)  # Dense
            elif i < middle_cutoff:
                ratios.append(0.5)  # Moderate sparse
            else:
                ratios.append(0.75)  # Light sparse
        description = "Dense-to-Sparse: Early=Dense, Middle=L/2, Late=3L/4"

    elif schedule == SparsitySchedule.AGGRESSIVE_MIDDLE:
        # Based on redundancy research: Middle layers most sparse
        # Early: Moderate (k=L/2)
        # Middle: Aggressive (k=L/4) - most redundant
        # Late: Moderate (k=L/2)
        ratios = []
        early_cutoff = n_layers // 3
        middle_cutoff = 2 * n_layers // 3

        for i in range(n_layers):
            if i < early_cutoff:
                ratios.append(0.5)  # Moderate
            elif i < middle_cutoff:
                ratios.append(0.25)  # Aggressive sparse
            else:
                ratios.append(0.5)  # Moderate
        description = "Aggressive-Middle: Early=L/2, Middle=L/4, Late=L/2"

    elif schedule == SparsitySchedule.PROGRESSIVE_SPARSE:
        # Original hypothesis: Dense foundation, aggressive middle, moderate late
        # Early: Dense (k=L)
        # Middle: Aggressive (k=L/4)
        # Late: Moderate (k=L/2)
        ratios = []
        early_cutoff = n_layers // 3
        middle_cutoff = 2 * n_layers // 3

        for i in range(n_layers):
            if i < early_cutoff:
                ratios.append(1.0)  # Dense
            elif i < middle_cutoff:
                ratios.append(0.25)  # Aggressive sparse
            else:
                ratios.append(0.5)  # Moderate
        description = "Progressive-Sparse: Early=Dense, Middle=L/4, Late=L/2"

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    # Compute actual k values
    k_values = [int(seq_len * ratio) for ratio in ratios]

    return LayerSparsityConfig(
        schedule_name=schedule.value,
        layer_k_values=k_values,
        layer_k_ratios=ratios,
        description=description
    )


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
        """
        batch_size, seq_len, _ = x.shape

        # Compute indexer queries: [batch, seq_len, indexer_heads, indexer_dim]
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.indexer_heads, self.indexer_dim)

        # Compute indexer keys: [batch, seq_len, indexer_dim]
        keys = self.k_proj(x)

        # Compute indexer weights: [batch, seq_len, indexer_heads]
        weights = self.w_proj(x)

        # Compute dot products: q_{t,j} · k_s for all t, s, j
        dots = torch.einsum('bthd,bsd->bths', queries, keys)

        # Apply ReLU activation
        activated = F.relu(dots)

        # Weight each head: w_{t,j} · ReLU(q_{t,j} · k_s)
        weighted = activated * weights.unsqueeze(-1)

        # Sum across heads: Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)
        index_scores = weighted.sum(dim=2)

        return index_scores


class AdaptiveTopKSelector(nn.Module):
    """
    Adaptive Top-K Token Selection with per-layer k values

    Args:
        default_top_k: Default k value (can be overridden per forward pass)
    """
    def __init__(self, default_top_k: int = 512):
        super().__init__()
        self.default_top_k = default_top_k

    def forward(
        self,
        index_scores: torch.Tensor,
        top_k: Optional[int] = None,
        apply_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Select top-k tokens based on index scores

        Args:
            index_scores: Index scores [batch, seq_len_q, seq_len_k]
            top_k: Number of tokens to select (overrides default)
            apply_causal_mask: Whether to apply causal masking

        Returns:
            - top_k_mask: Boolean mask [batch, seq_len_q, seq_len_k]
            - top_k_indices: Indices of selected tokens [batch, seq_len_q, k]
            - stats: Dictionary with selection statistics
        """
        batch_size, seq_len_q, seq_len_k = index_scores.shape

        # Use provided k or default
        k = top_k if top_k is not None else self.default_top_k

        # Apply causal mask: token t can only attend to tokens <= t
        if apply_causal_mask:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=index_scores.device),
                diagonal=1
            ).bool()
            index_scores = index_scores.masked_fill(causal_mask.unsqueeze(0), -1e9)

        # Select top-k indices for each query token
        actual_k = min(k, seq_len_k)
        top_k_values, top_k_indices = torch.topk(
            index_scores,
            k=actual_k,
            dim=-1,
            largest=True
        )

        # Create boolean mask from indices
        top_k_mask = torch.zeros_like(index_scores, dtype=torch.bool)
        top_k_mask.scatter_(2, top_k_indices, True)

        # Compute statistics
        sparsity = 1.0 - (top_k_mask.sum().item() / top_k_mask.numel())
        stats = {
            'sparsity': sparsity,
            'actual_k': actual_k,
            'k_ratio': actual_k / seq_len_k
        }

        return top_k_mask, top_k_indices, stats


class AdaptiveSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention with Adaptive Per-Layer Top-K

    Each layer can have a different sparsity level (k value) based on its
    functional role in the transformer hierarchy.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        layer_idx: Layer index (0-indexed)
        layer_top_k: Top-k value for this specific layer
        indexer_heads: Number of indexer heads
        indexer_dim: Dimension of indexer queries/keys
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        layer_idx: int,
        layer_top_k: int,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.layer_idx = layer_idx
        self.layer_top_k = layer_top_k

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

        # Adaptive token selector
        self.selector = AdaptiveTopKSelector(default_top_k=layer_top_k)

        # Whether to use sparse attention
        self.use_sparse = True

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with adaptive sparse attention

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_stats: Whether to return selection statistics

        Returns:
            - output: Attention output [batch_size, seq_len, d_model]
            - stats: Selection statistics if return_stats=True
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        stats = None

        if self.use_sparse:
            # Compute index scores
            index_scores = self.indexer(x)

            # Select top-k tokens (using layer-specific k)
            top_k_mask, top_k_indices, selector_stats = self.selector(
                index_scores,
                top_k=self.layer_top_k,
                apply_causal_mask=True
            )

            # Create attention mask
            attn_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len,
                device=x.device,
                dtype=Q.dtype
            )
            attn_mask = attn_mask.masked_fill(~top_k_mask.unsqueeze(1), float('-inf'))

            # Apply sparse attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0
            )

            if return_stats:
                stats = {
                    'layer_idx': self.layer_idx,
                    'layer_k': self.layer_top_k,
                    **selector_stats
                }
        else:
            # Dense attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )

            if return_stats:
                stats = {
                    'layer_idx': self.layer_idx,
                    'layer_k': seq_len,
                    'sparsity': 0.0,
                    'k_ratio': 1.0
                }

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)

        return output, stats

    def enable_sparse(self):
        """Enable sparse attention"""
        self.use_sparse = True

    def disable_sparse(self):
        """Disable sparse attention (use dense)"""
        self.use_sparse = False

    def update_layer_k(self, new_k: int):
        """Update the layer's top-k value dynamically"""
        self.layer_top_k = new_k
        self.selector.default_top_k = new_k


def print_schedule_info(config: LayerSparsityConfig, n_layers: int):
    """Print detailed information about a sparsity schedule"""
    print(f"\n{'='*80}")
    print(f"Sparsity Schedule: {config.schedule_name}")
    print(f"{'='*80}")
    print(f"Description: {config.description}")
    print(f"\nPer-Layer Configuration:")
    print(f"{'Layer':<10} {'k Ratio':<15} {'Function':<30}")
    print(f"{'-'*80}")

    for i in range(n_layers):
        ratio = config.layer_k_ratios[i] if i < len(config.layer_k_ratios) else config.layer_k_ratios[-1]

        # Categorize layer
        early_cutoff = n_layers // 3
        middle_cutoff = 2 * n_layers // 3
        if i < early_cutoff:
            function = "Early (local patterns)"
        elif i < middle_cutoff:
            function = "Middle (feature composition)"
        else:
            function = "Late (global context)"

        print(f"Layer {i:<4} {ratio:<15.2%} {function:<30}")
    print(f"{'='*80}\n")
