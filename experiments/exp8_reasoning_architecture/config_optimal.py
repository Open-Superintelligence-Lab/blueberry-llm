"""
Optimal Configuration based on Ablation Study
Best performing settings for Reasoning Architecture (Exp8)

Based on ablation study of 15 configurations:
- Best config: 4_high_lr with Val Loss 6.67, Perplexity 787
- Key findings: Higher LR, no warmup, no dropout for short training
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimalShortTrainingConfig:
    """
    Optimal config for short training runs (<1000 steps)
    Achieved Val Loss 6.67, Perplexity 787 in 100 steps
    """
    
    # Model Architecture (MoE)
    vocab_size: int = 50257
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    max_position_embeddings: int = 2048
    
    # MoE Configuration
    num_experts: int = 4
    expert_top_k: int = 2
    
    # MLP
    hidden_ratio: int = 4
    intermediate_size: Optional[int] = None
    
    # Regularization - REDUCED for short training
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0  # No dropout for short training!
    
    # Training - OPTIMIZED from ablation study
    batch_size: int = 16
    learning_rate: float = 6e-4  # 0.0006 - HIGHER LR wins!
    weight_decay: float = 0.1
    max_steps: int = 500  # Extended from 100
    warmup_steps: int = 0  # No warmup for short training!
    gradient_clip: float = 0.5  # Lower clip works better
    
    # Optimizer
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # Data
    max_seq_len: int = 512
    num_documents: int = 70_000
    max_tokens: int = 70_000_000
    
    # Evaluation
    eval_interval: int = 50
    eval_batches: int = 10
    
    # Logging
    log_interval: int = 10
    
    # Checkpointing
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints_optimal"
    
    # Device
    device: str = "cuda"
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * self.hidden_ratio


@dataclass
class OptimalLongTrainingConfig:
    """
    Optimal config for longer training runs (1000+ steps)
    Expected: Val Loss <4.0, Perplexity <50
    """
    
    # Model Architecture (MoE)
    vocab_size: int = 50257
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    max_position_embeddings: int = 2048
    
    # MoE Configuration
    num_experts: int = 4
    expert_top_k: int = 2
    
    # MLP
    hidden_ratio: int = 4
    intermediate_size: Optional[int] = None
    
    # Regularization - Light dropout for long training
    rms_norm_eps: float = 1e-6
    dropout: float = 0.05  # Small dropout for regularization
    
    # Training - Balanced for long runs
    batch_size: int = 16
    learning_rate: float = 3e-4  # 0.0003 - Stable for long training
    weight_decay: float = 0.1
    max_steps: int = 2000
    warmup_steps: int = 100  # Warmup helps for long training
    gradient_clip: float = 1.0
    
    # Optimizer
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # Data
    max_seq_len: int = 512
    num_documents: int = 70_000
    max_tokens: int = 70_000_000
    
    # Evaluation
    eval_interval: int = 100
    eval_batches: int = 20
    
    # Logging
    log_interval: int = 20
    
    # Checkpointing
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints_optimal"
    
    # Device
    device: str = "cuda"
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * self.hidden_ratio


def get_optimal_short_config():
    """Get optimal config for quick experiments (<1000 steps)"""
    return OptimalShortTrainingConfig()


def get_optimal_long_config():
    """Get optimal config for full training (1000+ steps)"""
    return OptimalLongTrainingConfig()

