"""
Configuration for Experiment 10: Attention Mechanism Ablation
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionAblationConfig:
    """Configuration for attention mechanism ablation experiment"""
    
    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    vocab_size: int = 32000
    max_seq_len: int = 1024
    dropout: float = 0.1
    
    # Attention mechanism type
    attention_type: str = "mha"  # mha, mla, gqa, mqa, sliding_window, sparse, linear, identity
    
    # Mechanism-specific parameters
    # For GQA
    n_kv_heads: Optional[int] = 2  # Number of KV heads for GQA
    
    # For MLA
    qk_rope_dim: int = 32
    qk_nope_dim: int = 32
    kv_lora_rank: int = 64
    v_dim: int = 32
    
    # For Sliding Window
    window_size: int = 512
    
    # For Sparse Attention
    indexer_heads: int = 4
    indexer_dim: int = 64
    sparse_top_k: int = 512
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    warmup_steps: int = 100
    eval_interval: int = 100
    log_interval: int = 50
    
    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    
    # Compute
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = False
    
    # Logging
    experiment_name: str = "attention_ablation"
    save_dir: str = "experiments/exp10_attention_mechanism_ablation/results"


# Predefined configurations for each attention type
def get_mha_config() -> AttentionAblationConfig:
    """Standard Multi-Head Attention (baseline)"""
    return AttentionAblationConfig(
        attention_type="mha",
        experiment_name="mha_baseline"
    )


def get_mla_config() -> AttentionAblationConfig:
    """Multi-Head Latent Attention"""
    return AttentionAblationConfig(
        attention_type="mla",
        qk_rope_dim=32,
        qk_nope_dim=32,
        kv_lora_rank=64,
        v_dim=32,
        experiment_name="mla"
    )


def get_gqa_config(n_kv_heads: int = 2) -> AttentionAblationConfig:
    """Grouped Query Attention"""
    return AttentionAblationConfig(
        attention_type="gqa",
        n_kv_heads=n_kv_heads,
        experiment_name=f"gqa_kv{n_kv_heads}"
    )


def get_mqa_config() -> AttentionAblationConfig:
    """Multi-Query Attention"""
    return AttentionAblationConfig(
        attention_type="mqa",
        experiment_name="mqa"
    )


def get_sliding_window_config(window_size: int = 512) -> AttentionAblationConfig:
    """Sliding Window Attention"""
    return AttentionAblationConfig(
        attention_type="sliding_window",
        window_size=window_size,
        experiment_name=f"sliding_w{window_size}"
    )


def get_sparse_config(sparse_top_k: int = 512) -> AttentionAblationConfig:
    """DeepSeek-style Sparse Attention"""
    return AttentionAblationConfig(
        attention_type="sparse",
        sparse_top_k=sparse_top_k,
        indexer_heads=4,
        indexer_dim=64,
        experiment_name=f"sparse_k{sparse_top_k}"
    )


def get_identity_config() -> AttentionAblationConfig:
    """Identity (no attention)"""
    return AttentionAblationConfig(
        attention_type="identity",
        experiment_name="identity_no_attn"
    )


def get_linear_config() -> AttentionAblationConfig:
    """Linear Attention via Gated DeltaNet"""
    return AttentionAblationConfig(
        attention_type="linear",
        experiment_name="linear_deltanet"
    )


# Comprehensive ablation - all configurations to test
ALL_CONFIGS = {
    "mha": get_mha_config,
    "mla": get_mla_config,
    "gqa_2": lambda: get_gqa_config(n_kv_heads=2),
    "gqa_1": lambda: get_gqa_config(n_kv_heads=1),  # Same as MQA
    "mqa": get_mqa_config,
    "sliding_512": lambda: get_sliding_window_config(window_size=512),
    "sliding_256": lambda: get_sliding_window_config(window_size=256),
    "sparse_512": lambda: get_sparse_config(sparse_top_k=512),
    "sparse_256": lambda: get_sparse_config(sparse_top_k=256),
    "identity": get_identity_config,
    "linear": get_linear_config,
}

