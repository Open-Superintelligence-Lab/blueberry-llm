"""
Experiment 5: DeepSeek MHLA with/without Sparse Attention

This experiment compares:
- Baseline: DeepSeek Multi-Head Latent Attention (dense)
- Experimental: DeepSeek MHLA + Sparse Attention (Lightning Indexer)

Tests whether sparse token selection improves DeepSeek's already-efficient
multi-head latent attention architecture.
"""

from .sparse_mhla_attention import (
    LightningIndexer,
    TopKTokenSelector,
    DeepSeekSparseMLHA
)

from .exp5_models import (
    SparseMLHATransformerBlock,
    BaselineMLHATransformerBlock,
    SparseMLHAMoELLM,
    BaselineMLHAMoELLM,
    create_sparse_model,
    create_baseline_model,
    count_parameters
)

__all__ = [
    # Sparse attention components
    'LightningIndexer',
    'TopKTokenSelector',
    'DeepSeekSparseMLHA',
    # Transformer blocks
    'SparseMLHATransformerBlock',
    'BaselineMLHATransformerBlock',
    # Full models
    'SparseMLHAMoELLM',
    'BaselineMLHAMoELLM',
    # Factory functions
    'create_sparse_model',
    'create_baseline_model',
    'count_parameters',
]

