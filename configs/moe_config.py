from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MoEModelConfig:
    # Model architecture
    d_model: int = 1536  # Updated for ~3B params
    n_heads: int = 12  # Updated for ~3B params
    n_layers: int = 26  # Updated for ~3B params
    d_ff: int = 4096  # Updated for ~3B params (~2.67x d_model)
    use_mla: bool = False
    qk_rope_dim: int | None = 32
    qk_nope_dim: int | None = 128
    kv_lora_rank: int | None = 64
    v_dim: int | None = 128
    batch_size: int = 8  # Reduced for 3B model memory efficiency
    max_steps: int = 10000  # Increased for better training

    # Training parameters
    gradient_accumulation_steps: int = 12  # Increased to maintain effective batch size
    muon_lr: float = 0.02  # Reduced for 3B model stability
    muon_momentum: float = 0.95  # Slightly increased for larger model
    adamw_lr: float = 0.003  # Reduced for 3B model stability
    warmup_ratio: float = 0.05

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 10
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # MoE specific parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
