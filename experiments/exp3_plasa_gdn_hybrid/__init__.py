<<<<<<< HEAD
"""
Experiment 3: Qwen3-Next with Per-Layer Adaptive Sparse Attention (PLASA)

Three variants:
1. Baseline: Standard Qwen3-Next
2. PLASA-Only: All attention replaced with Per-Layer Adaptive Sparse Attention
3. Hybrid: PLASA for full_attention, Gated DeltaNet for linear_attention

PLASA uses PROGRESSIVE_SPARSE schedule:
- Early layers: Dense (k=L)
- Middle layers: Aggressive sparse (k=L/4)
- Late layers: Moderate sparse (k=L/2)
"""

from .models import BaselineQwen3, PLASAQwen3, HybridQwen3, create_model
from .config import ExperimentConfig, SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG

__all__ = [
    'BaselineQwen3',
    'PLASAQwen3',
    'HybridQwen3',
    'create_model',
    'ExperimentConfig',
    'SMALL_CONFIG',
    'MEDIUM_CONFIG',
    'LARGE_CONFIG',
]

=======
"""
Experiment 3: Qwen3-Next with Per-Layer Adaptive Sparse Attention (PLASA)

Three variants:
1. Baseline: Standard Qwen3-Next
2. PLASA-Only: All attention replaced with Per-Layer Adaptive Sparse Attention
3. Hybrid: PLASA for full_attention, Gated DeltaNet for linear_attention

PLASA uses PROGRESSIVE_SPARSE schedule:
- Early layers: Dense (k=L)
- Middle layers: Aggressive sparse (k=L/4)
- Late layers: Moderate sparse (k=L/2)
"""

from .models import BaselineQwen3, PLASAQwen3, HybridQwen3, create_model
from .config import ExperimentConfig, SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG

__all__ = [
    'BaselineQwen3',
    'PLASAQwen3',
    'HybridQwen3',
    'create_model',
    'ExperimentConfig',
    'SMALL_CONFIG',
    'MEDIUM_CONFIG',
    'LARGE_CONFIG',
]

>>>>>>> e8f2fc47e63993a97bc8e16b3b827a7c313d98bf
