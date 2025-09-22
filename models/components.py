"""
Complex neural network components for LLM architectures.

This module provides higher-level components like attention mechanisms,
mixture of experts, and transformer blocks that can be composed into
full model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .layers import AdaptiveLinear, Rotary, AdaptiveLayerNorm, create_adaptive_linear
from system import SYSTEM_CONFIG


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with GPU-adaptive optimizations.
    
    This implementation uses adaptive linear layers and supports
    architecture-specific optimizations like custom attention scaling.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        max_seq_len: int, 
        dropout: float = 0.1, 
        use_fp8: bool = False,
        attention_scale: Optional[float] = None
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_fp8: Whether to use FP8 precision
            attention_scale: Custom attention scaling factor
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_fp8 = use_fp8
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # QKV projection using adaptive linear layers
        self.qkv = AdaptiveLinear(d_model, d_model * 3, bias=False, use_fp8=use_fp8)
        
        # Output projection with zero initialization (from reference implementation)
        self.w_o = create_adaptive_linear(d_model, d_model, bias=False, zero_init=True, use_fp8=use_fp8)
        
        # Rotary positional embedding
        self.rotary = Rotary(self.d_k, max_seq_len)
        
        # Dropout
        self.dropout = dropout
        
        # Attention scaling - use architecture-specific values
        if attention_scale is not None:
            self.attention_scale = attention_scale
        elif SYSTEM_CONFIG.architecture == "blackwell":
            # Optimized scaling for Blackwell (from reference implementation)
            self.attention_scale = 0.12
        else:
            # Standard scaling
            self.attention_scale = 1.0 / (self.d_k ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # QKV projection
        qkv = self.qkv(x)  # [batch_size, seq_len, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_heads, seq_len, d_k]
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Apply rotary positional embedding
        # Convert to [batch_size, seq_len, n_heads, d_k] for RoPE
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        # Scaled dot-product attention with custom scaling
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, 
            is_causal=True, 
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.attention_scale
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)
        
        return output


class Expert(nn.Module):
    """
    Single expert network for MoE layers.
    
    This is essentially a two-layer MLP with adaptive linear layers
    and modern activation functions.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout: float = 0.1, 
        use_fp8: bool = False,
        activation: str = "silu"
    ):
        """
        Initialize expert network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_fp8: Whether to use FP8 precision
            activation: Activation function ("silu", "gelu", "relu")
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        # Two-layer MLP with adaptive operations
        self.linear1 = AdaptiveLinear(d_model, d_ff, bias=False, use_fp8=use_fp8)
        self.linear2 = create_adaptive_linear(d_ff, d_model, bias=False, zero_init=True, use_fp8=use_fp8)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation function
        if activation == "silu":
            self.act_fn = F.silu
        elif activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "relu":
            self.act_fn = F.relu
        elif activation == "relu_squared":
            # ReliU^2 from reference implementation
            self.act_fn = lambda x: F.relu(x).square()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TopKRouter(nn.Module):
    """
    Top-K router for mixture of experts.
    
    Routes each token to the top-k most relevant experts based on
    a learned gating function.
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_experts: int, 
        top_k: int = 2, 
        noise_std: float = 0.1,
        use_fp8: bool = False
    ):
        """
        Initialize top-K router.
        
        Args:
            d_model: Model dimension
            num_experts: Number of experts
            top_k: Number of experts to route to
            noise_std: Standard deviation for exploration noise
            use_fp8: Whether to use FP8 precision
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Gating network
        self.gate = AdaptiveLinear(d_model, num_experts, bias=False, use_fp8=use_fp8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tuple of:
            - router_weights: Softmax weights for selected experts [batch_size, seq_len, top_k]
            - expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            - router_probs: Full probability distribution [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get full probability distribution (for load balancing loss)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, router_probs


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with adaptive operations.
    
    This implementation supports load balancing, capacity factors,
    and efficient expert routing.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01,
        use_fp8: bool = False,
        activation: str = "silu"
    ):
        """
        Initialize mixture of experts.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            num_experts: Number of experts
            top_k: Number of experts to route to per token
            dropout: Dropout probability
            load_balancing_weight: Weight for load balancing loss
            use_fp8: Whether to use FP8 precision
            activation: Activation function for experts
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Create experts with adaptive operations
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout, use_fp8=use_fp8, activation=activation) 
            for _ in range(num_experts)
        ])

        # Create router
        self.router = TopKRouter(d_model, num_experts, top_k, use_fp8=use_fp8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through mixture of experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tuple of:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing decisions
        router_weights, expert_indices, router_probs = self.router(x)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get weights for this expert
                mask_for_expert = (expert_indices == expert_idx)  # [batch, seq, top_k]
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)

                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        
        This encourages the router to distribute tokens evenly across experts.
        
        Args:
            router_probs: Full probability distribution [batch_size, seq_len, num_experts]
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            
        Returns:
            Load balancing loss scalar
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()

        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts

        return aux_loss * self.load_balancing_weight


class MoETransformerBlock(nn.Module):
    """
    Transformer block with mixture of experts.
    
    This combines multi-head attention with MoE feed-forward layers,
    including proper normalization and residual connections.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_fp8: bool = False,
        norm_type: str = "rms"
    ):
        """
        Initialize MoE transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            num_experts: Number of experts in MoE layer
            top_k: Number of experts to route to per token
            dropout: Dropout probability
            use_fp8: Whether to use FP8 precision
            norm_type: Type of normalization ("rms", "layer")
        """
        super().__init__()

        # Attention layer with adaptive operations
        self.attention = MultiHeadAttention(
            d_model, n_heads, max_seq_len, dropout, use_fp8=use_fp8
        )

        # MoE layer with adaptive operations
        self.feed_forward = MixtureOfExperts(
            d_model, d_ff, num_experts, top_k, dropout, use_fp8=use_fp8
        )

        # Normalization layers
        self.norm1 = AdaptiveLayerNorm(d_model, norm_type=norm_type)
        self.norm2 = AdaptiveLayerNorm(d_model, norm_type=norm_type)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tuple of:
            - output: Block output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing loss from MoE layer
        """
        # Self-attention with pre-normalization
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward with pre-normalization
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, aux_loss


class StandardTransformerBlock(nn.Module):
    """
    Standard transformer block without MoE.
    
    This provides a simpler alternative to MoE blocks for comparison
    or for layers where MoE is not desired.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_fp8: bool = False,
        norm_type: str = "rms",
        activation: str = "silu"
    ):
        """
        Initialize standard transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_fp8: Whether to use FP8 precision
            norm_type: Type of normalization
            activation: Activation function
        """
        super().__init__()

        # Attention layer
        self.attention = MultiHeadAttention(
            d_model, n_heads, max_seq_len, dropout, use_fp8=use_fp8
        )

        # Standard feed-forward layer
        self.feed_forward = Expert(
            d_model, d_ff, dropout, use_fp8=use_fp8, activation=activation
        )

        # Normalization layers
        self.norm1 = AdaptiveLayerNorm(d_model, norm_type=norm_type)
        self.norm2 = AdaptiveLayerNorm(d_model, norm_type=norm_type)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through standard transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with pre-normalization
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward with pre-normalization
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x
