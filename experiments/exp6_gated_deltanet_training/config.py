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


# Predefined configurations for different scales

def get_small_config():
    """Small config for quick testing (~2M params)"""
    return ExperimentConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        hidden_ratio=2,
        max_seq_len=128,
        batch_size=2,
        max_steps=1000,
        warmup_steps=50,
        num_documents=500,
        max_tokens=500_000,
    )


def get_medium_config():
    """Medium config - default (~15M params)"""
    return ExperimentConfig(
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        hidden_ratio=4,
        max_seq_len=256,
        batch_size=4,
        max_steps=2000,
        warmup_steps=100,
        num_documents=1000,
        max_tokens=2_000_000,
    )


def get_large_config():
    """Large config for full training (~60M params)"""
    return ExperimentConfig(
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        hidden_ratio=4,
        max_seq_len=512,
        batch_size=8,
        max_steps=5000,
        warmup_steps=250,
        num_documents=2000,
        max_tokens=5_000_000,
    )


def get_xlarge_config():
    """Extra large config (~200M params)"""
    return ExperimentConfig(
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        hidden_ratio=4,
        max_seq_len=1024,
        batch_size=4,
        max_steps=10000,
        warmup_steps=500,
        num_documents=5000,
        max_tokens=10_000_000,
    )


# Predefined configurations for specific GPUs

def get_rtx4090_optimized_config():
    """Optimized for RTX 4090 (24GB VRAM) - 1000 steps with no data repetition"""
    return ExperimentConfig(
        # Larger model to use more GPU
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Longer sequences and large batch for GPU saturation
        max_seq_len=1024,
        batch_size=32,
        
        # Training params - 1000 steps
        max_steps=1000,
        warmup_steps=100,  # 10% warmup
        learning_rate=1e-3,  # Best from 200-step ablation (val_loss=6.161)
        gradient_clip=1.0,
        
        # Data - NO REPETITION for 1000 steps
        # Tokens needed: 32 batch × 1024 seq × 1000 steps = 32,768,000 (32.8M)
        # With 2x safety margin = 65,536,000 (65.5M)
        num_documents=10_000,
        max_tokens=70_000_000,  # 70M tokens (2x safety margin)
        
        # Evaluation settings
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
    )


def get_h100_optimized_config():
    """Optimized for NVIDIA H100 (80GB HBM3) - tuned via LR ablation study"""
    return ExperimentConfig(
        # Model architecture - same as 4090 for comparison
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Sequence and batch configuration
        max_seq_len=1024,
        batch_size=120,  # Optimized for ~90% memory utilization on H100
        
        # Training params - LR from ablation study (tested 7 LRs, 1e-3 won)
        max_steps=10000,  # Extended training for production
        warmup_steps=1000,  # 10% warmup
        learning_rate=1e-3,  # Best from ablation: val_loss=6.671
        gradient_clip=1.0,
        
        # Data - sufficient for 10k steps without repetition
        # 120 batch × 1024 seq × 10000 steps = 1.23B tokens needed
        num_documents=250_000,  # Increased 125x for diversity
        max_tokens=2_000_000_000,  # 2B tokens (1.6x safety margin, ~8GB RAM)
        
        # Evaluation settings
        eval_interval=100,  # Eval every 100 steps
        eval_batches=20,
        log_interval=25,  # More frequent logging for 10k steps
    )


def get_h100_1k_checkpoint_config():
    """H100 config for 1000-step checkpoint (to be resumed later for full training)"""
    return ExperimentConfig(
        # Model architecture
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Sequence and batch configuration
        max_seq_len=1024,
        batch_size=120,
        
        # Training params - optimized for 1k checkpoint
        max_steps=1000,  # Initial checkpoint run
        warmup_steps=100,  # Quick warmup (10%)
        learning_rate=1e-3,  # Best LR from ablation
        gradient_clip=1.0,
        
        # Data - sufficient for 1k steps
        # 120 batch × 1024 seq × 1000 steps = 123M tokens needed
        num_documents=50_000,  # Increased 25x for diversity
        max_tokens=200_000_000,  # 200M tokens (1.6x safety margin, ~800MB RAM)
        
        # Evaluation settings
        eval_interval=100,
        eval_batches=20,
        log_interval=50,
    )


def get_h100_5k_config():
    """H100 config for 5000-step training with sufficient data"""
    return ExperimentConfig(
        # Model architecture
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Sequence and batch configuration
        max_seq_len=1024,
        batch_size=120,
        
        # Training params - 5k steps with proper warmup and decay
        max_steps=5000,
        warmup_steps=500,  # 10% warmup
        learning_rate=1e-3,  # Best LR from ablation
        gradient_clip=1.0,
        
        # Data - sufficient to avoid repetition
        # 120 batch × 1024 seq × 5000 steps = 614M tokens needed
        num_documents=200_000,  # Increased 100x to get more diverse data
        max_tokens=1_000_000_000,  # 1B tokens (1.6x safety margin, ~4GB RAM)
        
        # Evaluation settings
        eval_interval=100,
        eval_batches=20,
        log_interval=25,
        save_interval=500,
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
