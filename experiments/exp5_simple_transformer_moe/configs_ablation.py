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

LARGE_BATCH_001 = AblationConfig(
    name="large_batch_lr001",
    batch_size=104,      
    seq_len=256,
    lr=0.01,
    grad_accum=1,
    max_steps=100
)

LONG_SEQ_001 = AblationConfig(
    name="long_seq_lr001",
    batch_size=6,
    seq_len=4096,       
    lr=0.01,
    grad_accum=1,
    max_steps=100
)

BALANCED_001 = AblationConfig(
    name="balanced_lr001",
    batch_size=26,      
    seq_len=1024,
    lr=0.01,
    grad_accum=1,
    max_steps=100
)

LARGE_BATCH_002 = AblationConfig(
    name="large_batch_lr002",
    batch_size=104,      
    seq_len=256,
    lr=0.02,
    grad_accum=1,
    max_steps=100
)

LONG_SEQ_002 = AblationConfig(
    name="long_seq_lr002",
    batch_size=6,
    seq_len=4096,       
    lr=0.02,
    grad_accum=1,
    max_steps=100
)

BALANCED_002 = AblationConfig(
    name="balanced_lr002",
    batch_size=26,      
    seq_len=1024,
    lr=0.02,
    grad_accum=1,
    max_steps=100
)

# LARGE_BATCH_0005 = AblationConfig(
#     name="large_batch_lr0005",
#     batch_size=104,      
#     seq_len=256,
#     lr=0.005,
#     grad_accum=1,
#     max_steps=100
# )

# LONG_SEQ_0005 = AblationConfig(
#     name="long_seq_lr0005",
#     batch_size=6,
#     seq_len=4096,       
#     lr=0.005,
#     grad_accum=1,
#     max_steps=100
# )

# BALANCED_0005 = AblationConfig(
#     name="balanced_lr0005",
#     batch_size=26,      
#     seq_len=1024,
#     lr=0.005,
#     grad_accum=1,
#     max_steps=100
# )

LARGE_BATCH_003 = AblationConfig(
    name="large_batch_lr003",
    batch_size=104,      
    seq_len=256,
    lr=0.03,
    grad_accum=1,
    max_steps=100
)

LONG_SEQ_003 = AblationConfig(
    name="long_seq_lr003",
    batch_size=6,
    seq_len=4096,       
    lr=0.03,
    grad_accum=1,
    max_steps=100
)

BALANCED_003 = AblationConfig(
    name="balanced_lr003",
    batch_size=26,      
    seq_len=1024,
    lr=0.03,
    grad_accum=1,
    max_steps=100
)

LARGE_BATCH_004 = AblationConfig(
    name="large_batch_lr004",
    batch_size=104,      
    seq_len=256,
    lr=0.04,
    grad_accum=1,
    max_steps=100
)

LONG_SEQ_004 = AblationConfig(
    name="long_seq_lr004",
    batch_size=6,
    seq_len=4096,       
    lr=0.04,
    grad_accum=1,
    max_steps=100
)

BALANCED_004 = AblationConfig(
    name="balanced_lr004",
    batch_size=26,      
    seq_len=1024,
    lr=0.04,
    grad_accum=1,
    max_steps=100
)

# ============================================================================
# PART 2: Extended training with best LR (0.03) for 1000 steps
# ============================================================================

LARGE_BATCH_003_LONG = AblationConfig(
    name="large_batch_lr003_1000steps",
    batch_size=104,
    seq_len=256,
    lr=0.03,
    grad_accum=1,
    max_steps=1000
)

LONG_SEQ_003_LONG = AblationConfig(
    name="long_seq_lr003_1000steps",
    batch_size=6,
    seq_len=4096,
    lr=0.03,
    grad_accum=1,
    max_steps=1000
)

BALANCED_003_LONG = AblationConfig(
    name="balanced_lr003_1000steps",
    batch_size=26,
    seq_len=1024,
    lr=0.03,
    grad_accum=1,
    max_steps=1000
)


# Registry: add configs here to run them
CONFIGS = {
    # LR = 0.01 configs
    'large_batch_lr001': LARGE_BATCH_001,
    'long_seq_lr001': LONG_SEQ_001,
    'balanced_lr001': BALANCED_001,

    # LR = 0.02 configs
    'large_batch_lr002': LARGE_BATCH_002,
    'long_seq_lr002': LONG_SEQ_002,
    'balanced_lr002': BALANCED_002,

    # LR = 0.03 configs (NEW)
    'large_batch_lr003': LARGE_BATCH_003,
    'long_seq_lr003': LONG_SEQ_003,
    'balanced_lr003': BALANCED_003,
    
    # LR = 0.04 configs (NEW)
    'large_batch_lr004': LARGE_BATCH_004,
    'long_seq_lr004': LONG_SEQ_004,
    'balanced_lr004': BALANCED_004,
    
    # PART 2: Extended 1000-step runs with LR=0.03
    'large_batch_lr003_1000steps': LARGE_BATCH_003_LONG,
    'long_seq_lr003_1000steps': LONG_SEQ_003_LONG,
    'balanced_lr003_1000steps': BALANCED_003_LONG,
}

