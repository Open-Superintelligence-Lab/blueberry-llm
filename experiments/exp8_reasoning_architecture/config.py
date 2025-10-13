"""
Configuration for Reasoning Architecture Experiment
Based on MoE (Mixture of Experts) architecture

Architecture: 768d × 12L × 12H with MoE layers
Uses sparse expert activation for improved capacity and efficiency
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for Reasoning Architecture experiment (4090-optimized)"""
    
    # Model Architecture (reduced for 4090)
    vocab_size: int = 50257
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    max_position_embeddings: int = 2048
    
    # MoE specific parameters (reduced for 4090)
    num_experts: int = 4  # Number of expert networks per MoE layer
    expert_top_k: int = 2  # Number of experts to activate per token
    
    # MLP configuration
    hidden_ratio: int = 4  # MLP expansion ratio
    intermediate_size: Optional[int] = None
    
    # Regularization
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1
    
    # Training (optimized for 4090)
    batch_size: int = 16
    learning_rate: float = 3e-4  # 0.0003 - reduced for stability
    weight_decay: float = 0.1
    max_steps: int = 100
    warmup_steps: int = 20  # Increased from 10 for better stability
    gradient_clip: float = 1.0
    
    # Optimizer
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # Data
    max_seq_len: int = 512
    num_documents: int = 70_000
    max_tokens: int = 70_000_000
    
    # Evaluation
    eval_interval: int = 25
    eval_batches: int = 10
    
    # Logging
    log_interval: int = 10
    
    # Checkpointing
    save_interval: int = 100
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
    Base reasoning config using MoE (Mixture of Experts) architecture
    Optimized for RTX 4090 (24GB VRAM)
    
    Architecture:
    - 512 hidden dimensions (reduced from 768)
    - 8 transformer layers with MoE (reduced from 12)
    - 4 experts per layer, activating top-2 per token (reduced from 8)
    - Learning rate: 0.002 for stable MoE training
    """
    config = ExperimentConfig(
        # Model architecture - MoE (4090-friendly)
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        hidden_ratio=4,
        
        # MoE configuration (reduced)
        num_experts=4,
        expert_top_k=2,
        
        # Sequence and batch configuration (reduced for 4090)
        max_seq_len=512,
        batch_size=16,
        
        # Training params
        max_steps=1000,
        warmup_steps=100,
        learning_rate=3e-4,  # 0.0003 - reduced for stability
        gradient_clip=1.0,
        
        # Data
        num_documents=70_000,
        max_tokens=70_000_000,
        
        # Evaluation settings
        eval_interval=50,
        eval_batches=10,  # Reduced from 20
        log_interval=10,
    )
    
    return config


def get_extended_reasoning_config():
    """
    Extended training config for longer reasoning experiments
    Uses MoE architecture with more training steps
    """
    config = get_base_reasoning_config()
    config.max_steps = 5000
    config.warmup_steps = 500
    return config


def get_recursive_reasoning_config():
    """
    Recursive reasoning config with hierarchical cycles and ACT
    
    Adds recursive reasoning on top of the MoE architecture:
    - H_cycles: High-level reasoning iterations
    - L_cycles: Low-level reasoning iterations per H cycle
    - ACT: Adaptive compute time for dynamic depth
    """
    config = get_base_reasoning_config()
    
    # Add recursive reasoning parameters (reduced for stability)
    config.recursive = {
        'H_cycles': 2,  # High-level cycles (reduced from 3)
        'L_cycles': 2,  # Low-level cycles per H cycle (reduced from 3)
        'halt_max_steps': 3,  # Maximum reasoning steps (reduced from 5)
        'halt_exploration_prob': 0.1,  # Exploration for ACT learning
        'use_act': True,  # Enable Adaptive Compute Time
    }
    
    return config


# Alias for convenience
get_reasoning_config = get_base_reasoning_config

