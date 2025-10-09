"""
Configuration for Qwen3-Next Per-Layer Adaptive Sparse Attention (PLASA) Experiment

This experiment tests PLASA with PROGRESSIVE_SPARSE schedule:
- Early layers (0-33%): Dense attention (k=L)
- Middle layers (33-66%): Aggressive sparse (k=L/4)
- Late layers (66-100%): Moderate sparse (k=L/2)

Based on research showing middle layers are more functionally redundant.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for the PLASA experiment"""
    
    # Model architecture
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    
    # Qwen3-Next specific
    # Layer types: "full_attention" or "linear_attention"
    layer_types: List[str] = None  # Will be set in __post_init__
    
    # Linear attention (Gated DeltaNet) config
    linear_num_value_heads: int = 4
    linear_num_key_heads: int = 4
    linear_key_head_dim: int = 64
    linear_value_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    
    # MoE config (set to 0 for dense model)
    num_experts: int = 0
    expert_top_k: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = None
    
    # Per-Layer Adaptive Sparse Attention config
    # Note: sparse_top_k is overridden by PROGRESSIVE_SPARSE schedule
    # which sets per-layer k values based on layer depth
    indexer_heads: int = 4
    indexer_dim: int = 64
    sparse_top_k: int = 512  # Overridden by per-layer schedule
    
    # Training
    batch_size: int = 8
    max_seq_len: int = 512
    learning_rate: float = 3e-4
    max_steps: int = 5000
    warmup_steps: int = 500
    eval_interval: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data
    num_documents: int = 1000  # Number of documents to load
    max_tokens: int = 2_000_000  # Ensure enough tokens for training without repetition
    vocab_size: int = 50257  # Will be set from tokenizer
    
    # Logging
    log_interval: int = 100
    save_dir: str = "results"
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Set default layer types if not provided"""
        if self.layer_types is None:
            # Alternate pattern: full, full, linear, full, full, linear
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if (i + 1) % 3 == 0:
                    self.layer_types.append("linear_attention")
                else:
                    self.layer_types.append("full_attention")
        
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []


# Small config for quick testing
SMALL_CONFIG = ExperimentConfig(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=1024,
    max_seq_len=256,
    max_steps=1000,
    batch_size=4,
)

# Medium config for full experiment (reduced for testing)
MEDIUM_CONFIG = ExperimentConfig(
    hidden_size=256,  # Reduced from 512
    num_hidden_layers=4,  # Reduced from 6
    num_attention_heads=4,  # Reduced from 8
    num_key_value_heads=2,  # Reduced from 4
    intermediate_size=1024,  # Reduced from 2048
    max_seq_len=256,  # Reduced from 512
    max_steps=30,  # Testing with 30 steps
    batch_size=4,  # Reduced from 8
)

# Large config (if you have resources)
LARGE_CONFIG = ExperimentConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=6,
    intermediate_size=3072,
    max_seq_len=1024,
    max_steps=10000,
    batch_size=4,
)

