"""Configuration for Simple Transformer MoE Experiment"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SimpleTransformerConfig:
    """Simple transformer configuration with MoE"""
    
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    
    # MoE parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
    
    # Training parameters
    batch_size: int = 24
    max_steps: int = 20000
    gradient_accumulation_steps: int = 4
    
    # Optimizer parameters
    muon_lr: float = 0.01
    muon_momentum: float = 0.95
    adamw_lr: float = 0.001
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000
    
    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100
    
    # Regularization
    dropout: float = 0.1
    
    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000, 15000)
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

