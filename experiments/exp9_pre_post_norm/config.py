from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Model
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_layers: int = 4
    max_seq_len: int = 512
    num_experts: int = 4
    top_k: int = 2
    dropout: float = 0.1
    
    # Training
    batch_size: int = 8
    seq_len: int = 256
    num_steps: int = 500
    lr: float = 3e-4
    weight_decay: float = 0.01
    aux_loss_weight: float = 0.01
    
    # Logging
    log_interval: int = 50
    device: str = "cuda"

