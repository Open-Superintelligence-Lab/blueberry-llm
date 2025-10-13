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
    """Configuration for Reasoning Architecture experiment"""
    
    # Model Architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 2048
    
    # MoE specific parameters
    num_experts: int = 8  # Number of expert networks per MoE layer
    expert_top_k: int = 2  # Number of experts to activate per token
    
    # MLP configuration
    hidden_ratio: int = 4  # MLP expansion ratio
    intermediate_size: Optional[int] = None
    
    # Regularization
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1
    
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
    Base reasoning config using MoE (Mixture of Experts) architecture
    
    Architecture:
    - 768 hidden dimensions
    - 12 transformer layers with MoE
    - 8 experts per layer, activating top-2 per token
    - Learning rate: 0.002 for stable MoE training
    """
    config = ExperimentConfig(
        # Model architecture - MoE
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_ratio=4,
        
        # MoE configuration
        num_experts=8,
        expert_top_k=2,
        
        # Sequence and batch configuration
        max_seq_len=1024,
        batch_size=48,
        
        # Training params
        max_steps=1000,
        warmup_steps=100,
        learning_rate=2e-3,  # 0.002 - good for MoE training
        gradient_clip=1.0,
        
        # Data
        num_documents=70_000,
        max_tokens=70_000_000,
        
        # Evaluation settings
        eval_interval=50,
        eval_batches=20,
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

