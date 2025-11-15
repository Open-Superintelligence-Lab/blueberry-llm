"""
Experiment 4: 18B Parameter MoE Training

This module contains the implementation of an 18 billion parameter
Mixture of Experts language model optimized for NVIDIA B200 GPUs.
"""

from .config_18b import MoE18BConfig
from .models_18b import MoE18BLLM, MoETransformerBlock
from .trainer_18b import train_18b_model

__all__ = [
    'MoE18BConfig',
    'MoE18BLLM',
    'MoETransformerBlock',
    'train_18b_model',
]

