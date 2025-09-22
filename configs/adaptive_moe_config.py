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

    # GPU-adaptive parameters (optimized for T4)
    use_fp8: bool = False  # T4 doesn't support FP8
    use_adaptive_matmul: bool = True  # Use adaptive matmul operations
    
    # Single GPU training parameters
    use_megatron: bool = False  # Disabled for single T4 GPU
    tensor_parallel_size: int = 1  # Single GPU
    pipeline_parallel_size: int = 1  # Single GPU

    def __post_init__(self):
        """Post-initialization to validate and adapt configuration based on hardware."""
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Auto-detect optimal settings based on T4 GPU
        if SYSTEM_CONFIG.architecture == "t4":
            print("🚀 T4 GPU detected - using FP16 optimization")
            # T4 is optimized for FP16, disable FP8
            self.use_fp8 = False
        elif SYSTEM_CONFIG.has_bf16_support:
            print("🚀 BF16 acceleration enabled")
        else:
            print("📋 Using FP16 precision (T4 optimized)")
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


def get_t4_optimized_config() -> AdaptiveMoEModelConfig:
    """Get an optimized configuration for T4 GPU."""
    return AdaptiveMoEModelConfig(
        d_model=384,  # Balanced for T4 memory
        n_heads=8,    # Optimal for T4
        n_layers=6,  # Balanced for T4
        d_ff=1536,   # Balanced for T4
        batch_size=12,  # Optimized for T4 memory
        max_steps=2000,  # Good training length
        max_seq_len=1024,  # Good sequence length for T4
        num_experts=8,  # Good expert count for T4
        expert_top_k=2,
        gradient_accumulation_steps=3,  # Balanced for T4
        use_fp8=False,  # T4 doesn't support FP8
        use_adaptive_matmul=True,
        muon_lr=0.01,
        eval_every=250,
        eval_steps=50,
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
