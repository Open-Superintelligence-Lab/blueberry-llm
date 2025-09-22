"""
Basic neural network layers with GPU-adaptive optimizations.

This module provides fundamental building blocks for LLM architectures
that automatically adapt to different GPU capabilities.
"""

import torch
import torch.nn as nn
from typing import Optional
from system import SYSTEM_CONFIG
from ops.matmul import matmul, matmul_fp8

# Optional import for RoPE
try:
    from torchtune.modules import RotaryPositionalEmbeddings
    TORCHTUNE_AVAILABLE = True
except ImportError:
    TORCHTUNE_AVAILABLE = False


class AdaptiveLinear(nn.Module):
    """
    GPU-adaptive linear layer that automatically uses the best matmul kernel.
    
    This layer replaces nn.Linear with an implementation that leverages
    architecture-specific optimizations like FP8 on Blackwell GPUs.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        use_fp8: bool = False,
        init_method: str = "auto"
    ):
        """
        Initialize the adaptive linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
            use_fp8: Whether to use FP8 precision (only on supported hardware)
            init_method: Weight initialization method ("auto", "blackwell", "standard")
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8 and SYSTEM_CONFIG.has_fp8_support
        self.init_method = init_method
        
        # Weight parameter
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Bias parameter (optional)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # FP8 scaling factors (only used if FP8 is enabled)
        if self.use_fp8:
            self.register_parameter('x_s', nn.Parameter(torch.tensor(1.0)))
            self.register_parameter('w_s', nn.Parameter(torch.tensor(1.0)))
            self.register_parameter('grad_s', nn.Parameter(torch.tensor(1.0)))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using optimal scaling for the current architecture."""
        if self.init_method == "auto":
            # Auto-select based on architecture
            if SYSTEM_CONFIG.architecture == "blackwell":
                self._blackwell_init()
            else:
                self._standard_init()
        elif self.init_method == "blackwell":
            self._blackwell_init()
        elif self.init_method == "standard":
            self._standard_init()
        else:
            raise ValueError(f"Unknown init method: {self.init_method}")
        
        # Initialize bias to zero
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _blackwell_init(self):
        """Blackwell-optimized weight initialization."""
        # Improved initialization from the reference implementation
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        nn.init.uniform_(self.weight, -bound, bound)
    
    def _standard_init(self):
        """Standard weight initialization."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using GPU-adaptive matmul.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            
        Returns:
            Output tensor [batch_size, seq_len, out_features]
        """
        if self.use_fp8 and self.training:
            # Use FP8 matmul for training on Blackwell
            output = matmul_fp8(
                x, 
                self.weight, 
                self.x_s.item(), 
                self.w_s.item(), 
                self.grad_s.item()
            )
        else:
            # Use adaptive matmul (automatically selects best kernel)
            output = matmul(x, self.weight)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        fp8_str = ", FP8" if self.use_fp8 else ""
        bias_str = ", bias" if self.bias is not None else ""
        return f'in_features={self.in_features}, out_features={self.out_features}{bias_str}{fp8_str}'


class Rotary(nn.Module):
    """
    Rotary Positional Embedding (RoPE) layer.
    
    Wraps torchtune's RotaryPositionalEmbeddings with a consistent interface.
    """
    
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        """
        Initialize RoPE layer.
        
        Args:
            dim: Embedding dimension (should be head dimension)
            max_seq_len: Maximum sequence length
            base: Base for the frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        if not TORCHTUNE_AVAILABLE:
            raise ImportError("torchtune package is required for RoPE. Install with: pip install torchtune")
        
        self.rope = RotaryPositionalEmbeddings(
            dim=dim, 
            max_seq_len=max_seq_len, 
            base=base
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            Output tensor with RoPE applied
        """
        return self.rope(x)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}'


class AdaptiveEmbedding(nn.Module):
    """
    Adaptive embedding layer with optimal initialization.
    
    This embedding layer uses architecture-specific initialization
    and can be configured for different GPU architectures.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        padding_idx: Optional[int] = None,
        init_method: str = "auto"
    ):
        """
        Initialize adaptive embedding.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            padding_idx: Index of padding token
            init_method: Initialization method
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init_method = init_method
        
        self.embedding = nn.Embedding(
            vocab_size, 
            embed_dim, 
            padding_idx=padding_idx
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        if self.init_method == "auto":
            if SYSTEM_CONFIG.architecture == "blackwell":
                # Blackwell-optimized initialization
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            else:
                # Standard initialization
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        else:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, embed_dim]
        """
        return self.embedding(input_ids)


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive normalization layer that chooses between LayerNorm and RMSNorm.
    
    RMSNorm is often preferred for modern architectures as it's simpler
    and more efficient.
    """
    
    def __init__(
        self, 
        normalized_shape: int, 
        eps: float = 1e-5, 
        norm_type: str = "auto"
    ):
        """
        Initialize adaptive normalization.
        
        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability
            norm_type: Type of normalization ("auto", "rms", "layer")
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Choose normalization type
        if norm_type == "auto":
            # RMSNorm is generally better for modern LLMs
            self.norm_type = "rms"
        else:
            self.norm_type = norm_type
        
        if self.norm_type == "rms":
            self.norm = nn.RMSNorm(normalized_shape, eps=eps)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        return self.norm(x)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, type={self.norm_type}'


def create_adaptive_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    zero_init: bool = False,
    use_fp8: bool = False
) -> AdaptiveLinear:
    """
    Factory function to create adaptive linear layers with common configurations.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias
        zero_init: Whether to zero-initialize weights (useful for output projections)
        use_fp8: Whether to use FP8 precision
        
    Returns:
        Configured AdaptiveLinear layer
    """
    layer = AdaptiveLinear(
        in_features, 
        out_features, 
        bias=bias, 
        use_fp8=use_fp8
    )
    
    if zero_init:
        # Zero initialization for output projections (from reference implementation)
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    return layer
