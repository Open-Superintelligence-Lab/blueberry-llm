"""
Common utilities and helper functions for matmul operations.

This module contains shared functionality used across different
architecture-specific implementations.
"""

import torch
from typing import Tuple


def ensure_contiguous(x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure input tensors are contiguous for optimal performance.
    
    Args:
        x: Input tensor
        w: Weight tensor
        
    Returns:
        Tuple of contiguous tensors
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    
    return x, w


def get_optimal_dtype_for_architecture() -> torch.dtype:
    """
    Get the optimal data type for the current architecture.
    
    Returns:
        Optimal torch.dtype for the current GPU
    """
    from system import SYSTEM_CONFIG
    
    if SYSTEM_CONFIG.has_fp8_support:
        return torch.float8_e4m3fn
    elif SYSTEM_CONFIG.has_bf16_support:
        return torch.bfloat16
    elif SYSTEM_CONFIG.has_tensor_cores:
        return torch.float16
    else:
        return torch.float32


def validate_tensor_shapes(x: torch.Tensor, w: torch.Tensor) -> None:
    """
    Validate that tensor shapes are compatible for matmul.
    
    Args:
        x: Input tensor [..., d_model]
        w: Weight tensor [d_out, d_model]
        
    Raises:
        ValueError: If shapes are incompatible
    """
    if x.size(-1) != w.size(-1):
        raise ValueError(f"Incompatible shapes: x.size(-1)={x.size(-1)} != w.size(-1)={w.size(-1)}")
    
    if w.dim() != 2:
        raise ValueError(f"Weight tensor must be 2D, got {w.dim()}D")


def compute_output_shape(x: torch.Tensor, w: torch.Tensor) -> Tuple[int, ...]:
    """
    Compute the expected output shape for matmul(x, w).
    
    Args:
        x: Input tensor [..., d_model]
        w: Weight tensor [d_out, d_model]
        
    Returns:
        Expected output shape [..., d_out]
    """
    return (*x.shape[:-1], w.size(0))


def get_memory_usage_mb(x: torch.Tensor, w: torch.Tensor) -> float:
    """
    Estimate memory usage for matmul operation in MB.
    
    Args:
        x: Input tensor
        w: Weight tensor
        
    Returns:
        Estimated memory usage in MB
    """
    # Input + weight + output tensors
    input_bytes = x.numel() * x.element_size()
    weight_bytes = w.numel() * w.element_size()
    output_bytes = compute_output_shape(x, w)
    output_bytes = torch.tensor(output_bytes).numel() * x.element_size()
    
    total_bytes = input_bytes + weight_bytes + output_bytes
    return total_bytes / (1024 * 1024)  # Convert to MB


def should_use_chunked_matmul(x: torch.Tensor, w: torch.Tensor, threshold_mb: float = 1000.0) -> bool:
    """
    Determine if chunked matmul should be used based on memory usage.
    
    Args:
        x: Input tensor
        w: Weight tensor
        threshold_mb: Memory threshold in MB
        
    Returns:
        True if chunked matmul should be used
    """
    memory_mb = get_memory_usage_mb(x, w)
    return memory_mb > threshold_mb
