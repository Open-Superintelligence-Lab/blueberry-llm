"""
Model architectures and components for blueberry-llm.

This module provides modular neural network components that can be
composed to create different LLM architectures.
"""

from .layers import AdaptiveLinear, Rotary
from .components import MultiHeadAttention, Expert, TopKRouter, MixtureOfExperts, MoETransformerBlock
from .adaptive_llm import AdaptiveMoEMinimalLLM, create_model

__all__ = [
    # Basic layers
    'AdaptiveLinear',
    'Rotary',
    
    # Complex components
    'MultiHeadAttention',
    'Expert',
    'TopKRouter', 
    'MixtureOfExperts',
    'MoETransformerBlock',
    
    # Full models
    'AdaptiveMoEMinimalLLM',
    'create_model',
]
