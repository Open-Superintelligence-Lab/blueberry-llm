"""Configuration for 1.5B parameter model (optimal for $5 budget)"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SimpleTransformerConfig:
    """1.5B parameter transformer configuration with MoE"""
    
    # Model architecture - SCALED UP
    d_model: int = 96
    n_heads: int = 4  # Reduced to fit d_model
    n_layers: int = 20  # More layers for depth
    d_ff: int = 384
    
    # MoE parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
    
    # Training parameters - OPTIMIZED FOR BUDGET
    batch_size: int = 4  # Smaller batch for memory efficiency
    max_steps: int = 100000  # More steps with available time
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    
    # Optimizer parameters
    muon_lr: float = 0.01
    muon_momentum: float = 0.95
    adamw_lr: float = 0.001
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 200_000_000  # 200M tokens with $5 budget
    
    # Evaluation
    eval_every: int = 1000
    eval_steps: int = 100
    
    # Regularization
    dropout: float = 0.1
    
    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (10000, 25000, 50000, 75000)
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

