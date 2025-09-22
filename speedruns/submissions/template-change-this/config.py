"""
T4 Speedrun Configuration
"""

from dataclasses import dataclass
from configs import AdaptiveMoEModelConfig


@dataclass
class T4SpeedrunConfig(AdaptiveMoEModelConfig):
    """T4 Speedrun Configuration"""
    
    # Speedrun constraints
    SPEEDRUN_TIME_LIMIT_MINUTES: int = 5
    SPEEDRUN_DATASET_SEED: int = 42
    SPEEDRUN_VAL_SPLIT: float = 0.1
    SPEEDRUN_MAX_TOKENS: int = 100000
    SPEEDRUN_MAX_DOCUMENTS: int = 1000
    
    # Default configuration
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    batch_size: int = 16
    max_steps: int = 2000
    max_seq_len: int = 512
    num_documents: int = 1000
    max_tokens: int = 100000
    
    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01
    eval_every: int = 200
    eval_steps: int = 50
    
    # MoE parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
    
    # T4 optimizations
    use_amp: bool = True
    use_fp8: bool = False
    use_adaptive_matmul: bool = True
    
    def __post_init__(self):
        """Initialize configuration"""
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.num_documents == self.SPEEDRUN_MAX_DOCUMENTS
        assert self.max_tokens == self.SPEEDRUN_MAX_TOKENS
        self.use_fp8 = False  # T4 doesn't support FP8


def get_t4_speedrun_config() -> T4SpeedrunConfig:
    """Get default T4 speedrun configuration"""
    return T4SpeedrunConfig()


def create_custom_t4_config(**kwargs) -> T4SpeedrunConfig:
    """Create custom T4 speedrun configuration"""
    config = T4SpeedrunConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Unknown parameter: {key}")
    
    config.__post_init__()
    return config