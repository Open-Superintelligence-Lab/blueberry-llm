"""
Configuration for Reasoning Architecture Experiment
Based on Experiment 7's Hybrid Sparse 17% winner (val_loss=4.055)

Architecture: 768d × 12L × 12H
Attention: Layers [5, 11] (17% - mid and late positioning)
DeltaNet: Layers [0, 1, 2, 3, 4, 6, 7, 8, 9, 10] (83%)
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for Reasoning Architecture experiment"""
    
    # Model Architecture (from exp7 winner)
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 2048
    
    # DeltaNet specific
    expand_k: float = 1.0  # Key expansion ratio
    expand_v: float = 1.0  # Value expansion ratio
    
    # MLP configuration
    hidden_ratio: int = 4  # MLP expansion ratio
    intermediate_size: Optional[int] = None
    
    # Hybrid Model Configuration - WINNER from exp7
    # Attention on layers [5, 11] for 17% sparse attention
    attn_config: Optional[dict] = None
    
    # Regularization
    rms_norm_eps: float = 1e-6
    
    # Training (from exp7 winner)
    batch_size: int = 48
    learning_rate: float = 2e-3  # 0.002 - optimal for hybrids
    weight_decay: float = 0.1
    max_steps: int = 1000
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # Optimizer
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # Data
    max_seq_len: int = 1024
    num_documents: int = 70_000
    max_tokens: int = 70_000_000
    
    # Evaluation
    eval_interval: int = 50
    eval_batches: int = 20
    
    # Logging
    log_interval: int = 10
    
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


def get_base_reasoning_config():
    """
    Base reasoning config using exp7's winning Hybrid Sparse 17% architecture
    
    Winner stats from exp7:
    - Val loss: 4.055 (best of 13 architectures)
    - 27% better than pure Transformer
    - 8% better than pure DeltaNet
    - Throughput: 118K tokens/sec
    
    Architecture:
    - Attention on layers [5, 11] (mid and near-end)
    - DeltaNet on 10 other layers
    - Learning rate: 0.002 (hybrids need higher LR than pure DeltaNet)
    """
    config = ExperimentConfig(
        # Model architecture - WINNER from exp7
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # Sequence and batch configuration - from exp7 winner
        max_seq_len=1024,
        batch_size=48,
        
        # Training params
        max_steps=1000,
        warmup_steps=100,
        learning_rate=2e-3,  # 0.002 - optimal for hybrids from exp7
        gradient_clip=1.0,
        
        # Data - same as exp7
        num_documents=70_000,
        max_tokens=70_000_000,
        
        # Evaluation settings
        eval_interval=50,
        eval_batches=20,
        log_interval=10,
    )
    
    # CRITICAL: Set winning attention configuration
    # Attention at layers 5 (mid) and 11 (near-end)
    config.attn_config = {
        'layers': [5, 11],  # Winner configuration from exp7
        'window_size': 2048,
        'qkv_bias': False,
        'rope_theta': 10000.0,
    }
    
    return config


def get_extended_reasoning_config():
    """
    Extended training config for longer reasoning experiments
    Uses winning architecture with more training steps
    """
    config = get_base_reasoning_config()
    config.max_steps = 5000
    config.warmup_steps = 500
    return config


def get_recursive_reasoning_config():
    """
    Recursive reasoning config with hierarchical cycles and ACT
    
    Adds recursive reasoning on top of the winning architecture:
    - H_cycles: High-level reasoning iterations
    - L_cycles: Low-level reasoning iterations per H cycle
    - ACT: Adaptive compute time for dynamic depth
    """
    config = get_base_reasoning_config()
    
    # Add recursive reasoning parameters
    config.recursive = {
        'H_cycles': 3,  # High-level cycles
        'L_cycles': 3,  # Low-level cycles per H cycle
        'halt_max_steps': 5,  # Maximum reasoning steps
        'halt_exploration_prob': 0.1,  # Exploration for ACT learning
        'use_act': True,  # Enable Adaptive Compute Time
    }
    
    return config


# Alias for convenience
get_reasoning_config = get_base_reasoning_config

