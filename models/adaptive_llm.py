"""
Main adaptive LLM model architectures.

This module contains complete model implementations that compose
the basic layers and components into full LLM architectures.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .layers import AdaptiveEmbedding, AdaptiveLinear, AdaptiveLayerNorm, create_adaptive_linear
from .components import MoETransformerBlock, StandardTransformerBlock
from configs import AdaptiveMoEModelConfig
from system import SYSTEM_CONFIG


class AdaptiveMoEMinimalLLM(nn.Module):
    """
    GPU-adaptive MoE LLM with automatic optimization.
    
    This model automatically adapts to the available GPU hardware,
    using optimizations like FP8 precision on Blackwell GPUs.
    """
    
    def __init__(self, config: AdaptiveMoEModelConfig):
        """
        Initialize the adaptive MoE LLM.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Token embeddings with adaptive initialization
        self.token_embedding = AdaptiveEmbedding(
            config.vocab_size, 
            config.d_model,
            init_method="auto"
        )
        
        # Position dropout
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks with adaptive operations
        self.transformer_blocks = nn.ModuleList([
            MoETransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                num_experts=config.num_experts,
                top_k=config.expert_top_k,
                dropout=config.dropout,
                use_fp8=config.use_fp8
            )
            for i in range(config.n_layers)
        ])

        # Output layers
        self.norm = AdaptiveLayerNorm(config.d_model, norm_type="rms")
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head with adaptive operations
        # Tied with embeddings for parameter efficiency
        self.lm_head = create_adaptive_linear(
            config.d_model, 
            config.vocab_size, 
            bias=False, 
            zero_init=True,  # Zero init from reference implementation
            use_fp8=config.use_fp8
        )
        
        # Tie weights between embedding and output
        self.lm_head.weight = self.token_embedding.embedding.weight

        # Apply initialization
        self.apply(self._init_weights)
        
        # Print model info
        self._print_model_info()

    def _init_weights(self, module):
        """
        Initialize model weights with architecture-specific optimizations.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, AdaptiveLinear):
            # Weight initialization is handled in AdaptiveLinear
            pass
        elif isinstance(module, (AdaptiveEmbedding, nn.Embedding)):
            # Embedding initialization is handled in AdaptiveEmbedding
            pass
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # Standard normalization initialization
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)

    def _print_model_info(self):
        """Print model information and optimizations."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count active vs expert parameters
        active_params = 0
        expert_params = 0
        
        for name, param in self.named_parameters():
            if 'expert' in name:
                expert_params += param.numel()
            else:
                active_params += param.numel()
        
        print(f"\n📊 {self.__class__.__name__} Model Information:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Active parameters: {active_params:,}")
        print(f"   Expert parameters: {expert_params:,}")
        print(f"   Parameter efficiency: {active_params/total_params:.1%} active per forward pass")
        
        # GPU optimizations
        if self.config.use_fp8 and SYSTEM_CONFIG.has_fp8_support:
            print(f"   🚀 FP8 acceleration enabled")
        elif SYSTEM_CONFIG.has_bf16_support:
            print(f"   🚀 BF16 acceleration enabled")
        
        if self.config.use_adaptive_matmul:
            print(f"   ⚡ Adaptive matmul operations enabled")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            return_aux_loss: Whether to return auxiliary losses
            
        Returns:
            Tuple of:
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - aux_loss: Combined auxiliary losses (only if return_aux_loss=True)
        """
        # Token embeddings with scaling
        x = self.token_embedding(input_ids)
        x = x * math.sqrt(self.config.d_model)  # Scale embeddings
        x = self.position_dropout(x)

        # Collect auxiliary losses from MoE layers
        aux_losses = []

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x, aux_loss = block(x)
            if aux_loss is not None and return_aux_loss:
                aux_losses.append(aux_loss)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        # Combine auxiliary losses
        total_aux_loss = sum(aux_losses) if aux_losses else None

        if return_aux_loss:
            return logits, total_aux_loss
        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.embedding.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds
            
        Returns:
            MFU as a fraction of peak FLOPS
        """
        # First estimate the number of flops we do per iteration
        # See PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.d_model//cfg.n_heads, cfg.max_seq_len
        
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Express our flops throughput as ratio of A100 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        
        # A100 GPU bfloat16 peak flops is 312 TFLOPS
        flops_promised = 312e12
        if SYSTEM_CONFIG.architecture == "blackwell":
            # RTX 5090 has higher peak FLOPS
            flops_promised = 400e12  # Estimated
        
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(input_ids, return_aux_loss=False)
            
            # Get logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Truncate if sequence gets too long
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
        
        self.train()
        return input_ids


class AdaptiveStandardLLM(nn.Module):
    """
    Standard transformer LLM without MoE, but with GPU adaptations.
    
    This provides a simpler baseline model for comparison with MoE.
    """
    
    def __init__(self, config: AdaptiveMoEModelConfig):
        """
        Initialize the adaptive standard LLM.
        
        Args:
            config: Model configuration (MoE parameters will be ignored)
        """
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = AdaptiveEmbedding(
            config.vocab_size, 
            config.d_model,
            init_method="auto"
        )
        
        # Position dropout
        self.position_dropout = nn.Dropout(config.dropout)

        # Standard transformer blocks
        self.transformer_blocks = nn.ModuleList([
            StandardTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                use_fp8=config.use_fp8
            )
            for i in range(config.n_layers)
        ])

        # Output layers
        self.norm = AdaptiveLayerNorm(config.d_model, norm_type="rms")
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head
        self.lm_head = create_adaptive_linear(
            config.d_model, 
            config.vocab_size, 
            bias=False, 
            zero_init=True,
            use_fp8=config.use_fp8
        )
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.embedding.weight

        # Apply initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, AdaptiveLinear):
            pass  # Handled in AdaptiveLinear
        elif isinstance(module, (AdaptiveEmbedding, nn.Embedding)):
            pass  # Handled in AdaptiveEmbedding
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = x * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits


def should_use_megatron(config: AdaptiveMoEModelConfig) -> bool:
    """
    Determine if Megatron-LM should be used based on configuration and hardware.
    
    Args:
        config: Model configuration
        
    Returns:
        True if Megatron should be used, False for native backend
    """
    # Don't use Megatron if explicitly disabled
    if not config.use_megatron:
        return False
    
    # Only use Megatron with multiple GPUs
    if torch.cuda.device_count() <= 1:
        return False
    
    # Check if Megatron dependencies are available
    try:
        import megatron.core
        return True
    except ImportError:
        print("⚠️ Megatron-LM not available, falling back to native backend")
        return False


def create_model(config: AdaptiveMoEModelConfig, model_type: str = "moe") -> nn.Module:
    """
    Factory function to create different model types.
    
    Args:
        config: Model configuration
        model_type: Type of model to create ("moe", "standard")
        
    Returns:
        Model instance
    """
    # Check if we should use Megatron backend
    if should_use_megatron(config):
        from .megatron_wrapper import create_megatron_model
        return create_megatron_model(config, model_type)
    
    # Use native backend
    if model_type == "moe":
        return AdaptiveMoEMinimalLLM(config)
    elif model_type == "standard":
        return AdaptiveStandardLLM(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a model.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "model_type": type(model).__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
    }
    
    # Add MoE-specific info
    if hasattr(model, 'config') and hasattr(model.config, 'num_experts'):
        active_params = 0
        expert_params = 0
        
        for name, param in model.named_parameters():
            if 'expert' in name:
                expert_params += param.numel()
            else:
                active_params += param.numel()
        
        info.update({
            "active_params": active_params,
            "expert_params": expert_params,
            "active_params_M": active_params / 1e6,
            "expert_params_M": expert_params / 1e6,
            "parameter_efficiency": active_params / total_params,
        })
    
    return info
