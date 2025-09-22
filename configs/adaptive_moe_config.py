"""
Configuration for Adaptive Mixture of Experts model.

This module contains the configuration dataclass for the GPU-adaptive
MoE model with automatic optimization based on hardware capabilities.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from system import SYSTEM_CONFIG


@dataclass
class AdaptiveMoEModelConfig:
    """
    Configuration for the Adaptive Mixture of Experts model.
    
    This configuration automatically adapts to the detected GPU architecture
    and enables optimizations like FP8 training on supported hardware.
    """
    
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 1000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # MoE specific parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01

    # GPU-adaptive parameters
    use_fp8: bool = True  # Enable FP8 on supported hardware
    use_adaptive_matmul: bool = True  # Use adaptive matmul operations
    
    # Megatron-LM distributed training parameters
    use_megatron: bool = False  # Enable Megatron-LM for multi-GPU training (auto-detected)
    tensor_parallel_size: int = 1  # Tensor parallelism size (auto-detected)
    pipeline_parallel_size: int = 1  # Pipeline parallelism size (auto-detected)

    def __post_init__(self):
        """Post-initialization to validate and adapt configuration based on hardware."""
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Auto-detect optimal settings based on GPU
        if SYSTEM_CONFIG.has_fp8_support and self.use_fp8:
            print("🚀 FP8 acceleration enabled for Blackwell architecture")
            # Optimize batch size for FP8 memory savings
            if self.batch_size < 32 and SYSTEM_CONFIG.memory_gb > 20:
                self.batch_size = min(32, self.batch_size * 2)
                print(f"   📈 Increased batch size to {self.batch_size} for FP8 efficiency")
        elif SYSTEM_CONFIG.has_bf16_support:
            print("🚀 BF16 acceleration enabled")
        else:
            print("📋 Using standard precision")
            # Disable FP8 if not supported
            self.use_fp8 = False
        
        # Note: Megatron is disabled by default. Use --use-megatron flag to enable.
        # Auto-detect Megatron usage based on GPU count
        # if SYSTEM_CONFIG.device_count > 1 and not hasattr(self, '_megatron_auto_detected'):
        #     self._megatron_auto_detected = True
        #     if not self.use_megatron:  # Only auto-enable if not explicitly set
        #         self.use_megatron = True
        #         # Auto-detect optimal parallelism sizes
        #         self.tensor_parallel_size = min(SYSTEM_CONFIG.device_count, 8)  # Max 8 for tensor parallel
        #         print(f"🚀 Megatron-LM auto-enabled for {SYSTEM_CONFIG.device_count} GPUs")
        #         print(f"   📊 Tensor parallel size: {self.tensor_parallel_size}")
    
    def get_optimal_dtype(self):
        """Get the optimal data type for this configuration."""
        return SYSTEM_CONFIG.get_optimal_dtype()
    
    def supports_feature(self, feature: str) -> bool:
        """Check if the current configuration supports a specific feature."""
        feature_map = {
            "fp8": self.use_fp8 and SYSTEM_CONFIG.has_fp8_support,
            "adaptive_matmul": self.use_adaptive_matmul,
            "tensor_cores": SYSTEM_CONFIG.has_tensor_cores,
            "bf16": SYSTEM_CONFIG.has_bf16_support,
        }
        return feature_map.get(feature, False)
    
    def get_info(self):
        """Get a dictionary with configuration information."""
        return {
            "model_type": "AdaptiveMoE",
            "architecture": f"{self.d_model}d-{self.n_layers}L-{self.n_heads}H",
            "moe": f"{self.num_experts}experts-top{self.expert_top_k}",
            "training": f"{self.max_steps}steps-bs{self.batch_size}",
            "gpu_features": {
                "fp8": self.supports_feature("fp8"),
                "adaptive_matmul": self.supports_feature("adaptive_matmul"),
                "tensor_cores": self.supports_feature("tensor_cores"),
            },
            "optimal_dtype": str(self.get_optimal_dtype()),
        }
    
    def print_config(self):
        """Print a formatted configuration summary."""
        info = self.get_info()
        print(f"\n📋 {info['model_type']} Configuration:")
        print(f"   Architecture: {info['architecture']}")
        print(f"   MoE: {info['moe']}")
        print(f"   Training: {info['training']}")
        print(f"   Data: {self.max_tokens:,} tokens, seq_len {self.max_seq_len}")
        print(f"   GPU Features:")
        for feature, enabled in info['gpu_features'].items():
            status = "✅" if enabled else "❌"
            print(f"     {status} {feature}")
        print(f"   Optimal Dtype: {info['optimal_dtype']}")


def get_rtx4090_config() -> AdaptiveMoEModelConfig:
    """Get an optimized configuration for RTX 4090."""
    return AdaptiveMoEModelConfig(
        d_model=768,  # Increased from 384
        n_heads=12,   # Increased from 8
        n_layers=12,  # Increased from 6
        d_ff=3072,    # Increased from 1536
        batch_size=24,  # Increased from 16
        max_steps=2000,  # Increased from 1000
        max_seq_len=2048,  # Increased from 1024
        num_experts=16,  # Increased from 8
        expert_top_k=2,
        gradient_accumulation_steps=2,  # Reduced for larger batch
        use_fp8=False,  # RTX 4090 doesn't support FP8
        use_adaptive_matmul=True,
        muon_lr=0.01,
        eval_every=250,
        eval_steps=50,
    )


def get_rtx5090_config() -> AdaptiveMoEModelConfig:
    """Get an optimized configuration for RTX 5090."""
    return AdaptiveMoEModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        batch_size=32,  # Take advantage of 32GB VRAM
        max_steps=2000,
        max_seq_len=1024,  # Longer sequences
        num_experts=16,  # More experts for better capacity
        expert_top_k=2,
        use_fp8=True,  # Enable FP8 for Blackwell
        use_adaptive_matmul=True,
    )


def get_development_config() -> AdaptiveMoEModelConfig:
    """Get a fast configuration for development and testing."""
    return AdaptiveMoEModelConfig(
        d_model=256,  # 256 is divisible by 16
        n_heads=4,
        n_layers=4,
        d_ff=1024,   # 1024 is divisible by 16
        batch_size=16,
        max_steps=500,
        max_seq_len=256,
        num_experts=8,  # Changed from 4 to 8 (8 is divisible by 16)
        expert_top_k=2,
        eval_every=100,
        eval_steps=50,
    )
