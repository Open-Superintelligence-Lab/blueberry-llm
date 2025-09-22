"""
Hopper-specific matmul implementations optimized for H100.

This module contains matmul kernels optimized for Hopper architecture
with BF16 support and advanced tensor cores.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def matmul_bf16_tensor_cores(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    BF16 matmul optimized for Hopper (H100) tensor cores.
    
    Uses BF16 precision with Hopper's advanced tensor core acceleration.
    This is the optimal precision for H100 GPUs.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        w: Weight tensor [d_model, d_out]
        
    Returns:
        Output tensor [batch_size, seq_len, d_out]
    """
    # Ensure BF16 precision for tensor cores
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = w.to(torch.bfloat16)
    
    # Standard matmul - PyTorch automatically uses Hopper tensor cores
    return torch.matmul(x_bf16, w_bf16.T)


def matmul_fp16_tensor_cores(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    FP16 matmul using Hopper tensor cores.
    
    Alternative implementation for cases where BF16 is not desired.
    """
    # Ensure FP16 precision
    x_fp16 = x.to(torch.float16)
    w_fp16 = w.to(torch.float16)
    
    # Standard matmul with tensor core acceleration
    return torch.matmul(x_fp16, w_fp16.T)


def matmul_mixed_precision(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Mixed precision matmul optimized for Hopper.
    
    Uses FP16 for weights and BF16 for activations, which can provide
    good performance with numerical stability.
    """
    # Mixed precision: BF16 activations, FP16 weights
    x_bf16 = x.to(torch.bfloat16)
    w_fp16 = w.to(torch.float16)
    
    # Compute in BF16
    return torch.matmul(x_bf16, w_fp16.T.to(torch.bfloat16))


def matmul_with_gradient_checkpointing(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Matmul with gradient checkpointing for memory efficiency.
    
    Useful for very large models where memory is a constraint.
    """
    def matmul_fn(x_in, w_in):
        return torch.matmul(x_in.to(torch.bfloat16), w_in.to(torch.bfloat16).T)
    
    # Use gradient checkpointing to save memory
    return torch.utils.checkpoint.checkpoint(matmul_fn, x, w)


def get_optimal_precision(x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.dtype, torch.dtype]:
    """
    Determine optimal precision for Hopper architecture.
    
    Returns the best input and weight dtypes for this specific
    tensor configuration on Hopper.
    """
    # For Hopper, BF16 is generally optimal for activations
    # FP16 can be better for weights in some cases
    input_dtype = torch.bfloat16
    weight_dtype = torch.float16
    
    return input_dtype, weight_dtype
