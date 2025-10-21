from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Model
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    max_seq_len: int = 512
    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.1
    
    # Training
    batch_size: int = 8
    seq_len: int = 512
    num_steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    gradient_clip: float = 1.0
    aux_loss_weight: float = 0.01
    
    # Data
    num_documents: int = 1000
    max_tokens: int = 2_000_000
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 200
    device: str = "cuda"

