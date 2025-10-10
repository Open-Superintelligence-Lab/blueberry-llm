"""
Simple Transformer Configuration for MoE Experiments

This config is compatible with the global MoEModelConfig and provides
a simple interface for transformer experiments.
"""

from dataclasses import dataclass
from typing import Optional
import sys
import os

# Add project root to path to import MoEModelConfig
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from configs.moe_config import MoEModelConfig


@dataclass
class SimpleTransformerConfig:
    """Simple configuration for transformer experiments"""
    
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    
    # Training parameters
    batch_size: int = 24
    max_seq_len: int = 512
    max_steps: int = 20
    gradient_accumulation_steps: int = 4
    
    # Optimizer parameters
    muon_lr: float = 0.01
    muon_momentum: float = 0.95
    adamw_lr: float = 0.001
    weight_decay: float = 0.1
    
    # Data parameters
    num_documents: int = 2000
    max_tokens: int = 500000
    vocab_size: Optional[int] = None
    
    # Evaluation
    eval_every: int = 10
    eval_steps: int = 100
    
    # Regularization
    dropout: float = 0.1
    grad_clip: float = 1.0
    
    # Technical
    use_amp: bool = True
    
    # MoE parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
    
    def __post_init__(self):
        """Post-initialization validation"""
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
    
    def to_moe_config(self) -> MoEModelConfig:
        """Convert to MoEModelConfig for compatibility with global models"""
        return MoEModelConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            max_steps=self.max_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            muon_lr=self.muon_lr,
            muon_momentum=self.muon_momentum,
            adamw_lr=self.adamw_lr,
            weight_decay=self.weight_decay,
            num_documents=self.num_documents,
            max_tokens=self.max_tokens,
            vocab_size=self.vocab_size,
            eval_every=self.eval_every,
            eval_steps=self.eval_steps,
            dropout=self.dropout,
            grad_clip=self.grad_clip,
            use_amp=self.use_amp,
            num_experts=self.num_experts,
            expert_top_k=self.expert_top_k,
            load_balancing_weight=self.load_balancing_weight,
        )

