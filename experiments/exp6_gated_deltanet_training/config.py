"""
Configuration for Gated DeltaNet Training Experiment using FLA
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for FLA DeltaNet experiment"""
    
    # Model Architecture
    vocab_size: int = 50257
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    max_position_embeddings: int = 2048
    
    # DeltaNet specific
    expand_k: float = 1.0  # Key expansion ratio
    expand_v: float = 1.0  # Value expansion ratio
    
    # MLP configuration
    hidden_ratio: int = 4  # MLP expansion ratio (intermediate_size = hidden_size * hidden_ratio)
    intermediate_size: Optional[int] = None  # If None, will use hidden_size * hidden_ratio
    
    # Regularization
    rms_norm_eps: float = 1e-6
    
    # Training
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 2000
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # Optimizer
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # Data
    max_seq_len: int = 256
    num_documents: int = 1000
    max_tokens: int = 2_000_000
    
    # Evaluation
    eval_interval: int = 100
    eval_batches: int = 50
    
    # Logging
    log_interval: int = 50
    
    # Checkpointing
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda"
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        """Set intermediate size if not provided"""
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * self.hidden_ratio


# Predefined configurations for specific GPUs

def get_rtx4090_optimized_config():
    """Optimized for RTX 4090 (24GB VRAM) - maximize GPU utilization"""
    return ExperimentConfig(
        # Larger model to use more GPU
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Longer sequences and large batch for GPU saturation
        max_seq_len=1024,  # 4x longer than default
        batch_size=32,     # 8x larger than default!
        
        # Training params
        max_steps=2000,
        warmup_steps=200,
        learning_rate=3e-4,
        gradient_clip=1.0,
        
        # Data - more tokens for better training
        num_documents=2000,
        max_tokens=5_000_000,
        
        # More frequent evaluation since steps are more expensive
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
    )


def get_b200_optimized_config():
    """Optimized for NVIDIA B200 (190GB HBM3e) - same model as 4090, larger batch"""
    return ExperimentConfig(
        # Same model architecture as 4090 for easy comparison
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Same sequence length as 4090
        max_seq_len=1024,
        
        # 8x larger batch size to utilize ~8x more memory (190GB vs 24GB)
        batch_size=256,  # 8x the 4090's batch size
        
        # Training params - scale learning rate with sqrt(batch_size_ratio)
        # 3e-4 * sqrt(8) ≈ 8.5e-4
        max_steps=2000,
        warmup_steps=200,
        learning_rate=8.5e-4,  # Sqrt scaling: 3e-4 * sqrt(8)
        gradient_clip=1.0,
        
        # Data - same as 4090
        num_documents=2000,
        max_tokens=5_000_000,
        
        # Evaluation settings
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
    )
