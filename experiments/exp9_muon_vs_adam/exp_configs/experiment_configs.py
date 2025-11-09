"""
Experiment configurations to compare Muon vs Adam optimizers
"""
from dataclasses import dataclass, asdict
from typing import Optional, Literal
from configs.moe_config import MoEModelConfig


@dataclass
class ExperimentConfig:
    """Configuration for optimizer comparison experiments"""
    name: str
    description: str
    
    # Optimizer selection
    optimizer_type: Literal["muon", "adam", "muon_hybrid"] = "muon_hybrid"
    
    # Base model config overrides
    max_steps: int = 1000
    batch_size: int = 24
    
    # Learning rate configurations
    use_lr_schedule: bool = True
    lr_schedule_type: Literal["cosine", "constant", "step", "linear_decay"] = "cosine"
    warmup_steps_ratio: float = 0.05  # Fraction of total steps
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of initial LR
    
    # Optimizer-specific learning rates
    muon_lr: float = 0.01
    adam_lr: float = 0.001
    adamw_lr: float = 0.001  # For embeddings/norms in hybrid mode
    
    # Early stopping
    use_early_stopping: bool = False
    early_stopping_patience: int = 50  # Number of eval steps to wait
    early_stopping_min_delta: float = 0.001  # Minimum change to count as improvement
    
    # MoE specific
    load_balancing_weight: float = 0.01
    num_experts: int = 8
    expert_top_k: int = 2
    
    # Regularization
    dropout: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Evaluation
    eval_every: int = 10
    
    def to_moe_config(self, base_config: MoEModelConfig) -> MoEModelConfig:
        """Convert experiment config to MoEModelConfig"""
        config = MoEModelConfig(
            # Copy base architecture
            d_model=base_config.d_model,
            n_heads=base_config.n_heads,
            n_layers=base_config.n_layers,
            d_ff=base_config.d_ff,
            use_mla=base_config.use_mla,
            qk_rope_dim=base_config.qk_rope_dim,
            qk_nope_dim=base_config.qk_nope_dim,
            kv_lora_rank=base_config.kv_lora_rank,
            v_dim=base_config.v_dim,
            
            # Override with experiment settings
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            load_balancing_weight=self.load_balancing_weight,
            num_experts=self.num_experts,
            expert_top_k=self.expert_top_k,
            dropout=self.dropout,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            eval_every=self.eval_every,
            
            # Optimizer LRs
            muon_lr=self.muon_lr,
            adamw_lr=self.adamw_lr,
            
            # Keep other settings from base
            gradient_accumulation_steps=base_config.gradient_accumulation_steps,
            muon_momentum=base_config.muon_momentum,
            max_seq_len=base_config.max_seq_len,
            num_documents=base_config.num_documents,
            max_tokens=base_config.max_tokens,
            eval_steps=base_config.eval_steps,
            use_amp=base_config.use_amp,
            vocab_size=base_config.vocab_size,
            log_milestones=base_config.log_milestones,
        )
        return config


# Define experiments to compare Muon vs Adam
EXPERIMENTS = {
    "muon_baseline": ExperimentConfig(
        name="muon_baseline",
        description="Baseline: Hybrid Muon (2D weights) + AdamW (embeddings/norms)",
        optimizer_type="muon_hybrid",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        muon_lr=0.01,
        adamw_lr=0.001,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "adam_baseline": ExperimentConfig(
        name="adam_baseline",
        description="Pure Adam: AdamW for all parameters",
        optimizer_type="adam",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        adam_lr=0.001,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "adam_higher_lr": ExperimentConfig(
        name="adam_higher_lr",
        description="Adam with higher learning rate (0.002)",
        optimizer_type="adam",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        adam_lr=0.002,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "adam_lower_lr": ExperimentConfig(
        name="adam_lower_lr",
        description="Adam with lower learning rate (0.0005)",
        optimizer_type="adam",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        adam_lr=0.0005,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "muon_only": ExperimentConfig(
        name="muon_only",
        description="Pure Muon: Muon for all 2D parameters, minimal AdamW",
        optimizer_type="muon_hybrid",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        muon_lr=0.02,
        adamw_lr=0.001,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "muon_constant_lr": ExperimentConfig(
        name="muon_constant_lr",
        description="Muon with constant learning rate (no schedule)",
        optimizer_type="muon_hybrid",
        max_steps=1000,
        use_lr_schedule=False,
        lr_schedule_type="constant",
        muon_lr=0.01,
        adamw_lr=0.001,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "adam_constant_lr": ExperimentConfig(
        name="adam_constant_lr",
        description="Adam with constant learning rate (no schedule)",
        optimizer_type="adam",
        max_steps=1000,
        use_lr_schedule=False,
        lr_schedule_type="constant",
        adam_lr=0.001,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
}


def get_experiment(name: str) -> ExperimentConfig:
    """Get experiment configuration by name"""
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")
    return EXPERIMENTS[name]


def list_experiments():
    """Print all available experiments"""
    print("Available experiments:")
    print("=" * 80)
    for name, config in EXPERIMENTS.items():
        print(f"\n{name}:")
        print(f"  {config.description}")
        print(f"  - Optimizer: {config.optimizer_type}")
        print(f"  - Steps: {config.max_steps}")
        if config.optimizer_type == "adam":
            print(f"  - Adam LR: {config.adam_lr}")
        elif config.optimizer_type == "muon_hybrid":
            print(f"  - Muon LR: {config.muon_lr}, AdamW LR: {config.adamw_lr}")
        print(f"  - LR schedule: {config.lr_schedule_type if config.use_lr_schedule else 'none'}")
        print(f"  - Load balancing weight: {config.load_balancing_weight}")
        print(f"  - Early stopping: {config.use_early_stopping}")

