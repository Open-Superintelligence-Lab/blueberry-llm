"""
Editable configs for different ablation experiments.
Modify these directly to test different memory configurations.
"""

from dataclasses import dataclass

@dataclass
class AblationConfig:
    name: str
    batch_size: int
    seq_len: int
    lr: float
    grad_accum: int = 1
    max_steps: int = 20
    
    @property
    def effective_batch(self):
        return self.batch_size * self.grad_accum
    
    @property
    def tokens_per_step(self):
        return self.effective_batch * self.seq_len


# Edit these configs to max out your GPU memory!

LARGE_BATCH = AblationConfig(
    name="large_batch",
    batch_size=104,      # ← Edit: increase to use more memory
    seq_len=256,
    lr=0.01,
    grad_accum=1,
    max_steps=50
)

LONG_SEQ = AblationConfig(
    name="long_seq", 
    batch_size=6,
    seq_len=4096,       # ← Edit: increase for longer context
    lr=0.01,
    grad_accum=1,
    max_steps=50
)

BALANCED = AblationConfig(
    name="balanced",
    batch_size=26,      # ← Edit: balance between batch & seqlen
    seq_len=1024,
    lr=0.01,
    grad_accum=1,
    max_steps=50
)

CUSTOM = AblationConfig(
    name="custom",
    batch_size=32,      # ← Edit freely
    seq_len=384,
    lr=0.015,
    grad_accum=1,
    max_steps=50
)

# Quick test config (fast iteration)
QUICK_TEST = AblationConfig(
    name="quick_test",
    batch_size=16,
    seq_len=256,
    lr=0.01,
    grad_accum=1,
    max_steps=5
)

# Maximum memory configs (adjust based on your GPU)
# RTX 4090 (24GB) example values:
MAX_BATCH = AblationConfig(
    name="max_batch",
    batch_size=128,     # ← Push batch size limit
    seq_len=128,
    lr=0.02,
    grad_accum=1,
    max_steps=50
)

MAX_SEQ = AblationConfig(
    name="max_seq",
    batch_size=4,
    seq_len=2048,       # ← Push sequence length limit
    lr=0.005,
    grad_accum=1,
    max_steps=50
)


# Registry: add configs here to run them
CONFIGS = {
    'large_batch': LARGE_BATCH,
    'long_seq': LONG_SEQ,
    'balanced': BALANCED,
    'custom': CUSTOM,
    'quick': QUICK_TEST,
    'max_batch': MAX_BATCH,
    'max_seq': MAX_SEQ,
}

