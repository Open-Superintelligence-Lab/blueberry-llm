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
    
    def to_moe_config(self):
        """Convert to MoEModelConfig for compatibility with global models"""
        from configs.moe_config import MoEModelConfig
        return MoEModelConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            muon_lr=self.muon_lr,
            muon_momentum=self.muon_momentum,
            adamw_lr=self.adamw_lr,
            max_seq_len=self.max_seq_len,
            num_documents=self.num_documents,
            max_tokens=self.max_tokens,
            eval_every=self.eval_every,
            eval_steps=self.eval_steps,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
            grad_clip=self.grad_clip,
            use_amp=self.use_amp,
            vocab_size=self.vocab_size,
            log_milestones=self.log_milestones,
            num_experts=self.num_experts,
            expert_top_k=self.expert_top_k,
            load_balancing_weight=self.load_balancing_weight
        )

