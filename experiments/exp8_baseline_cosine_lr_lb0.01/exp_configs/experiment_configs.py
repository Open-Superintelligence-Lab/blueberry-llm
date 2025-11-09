"""
Experiment configurations to systematically test validation loss issues
"""
from dataclasses import dataclass, asdict
from typing import Optional, Literal
from configs.moe_config import MoEModelConfig


@dataclass
class ExperimentConfig:
    """Configuration for a systematic experiment"""
    name: str
    description: str
    
    # Base model config overrides
    max_steps: int = 1000
    batch_size: int = 24
    
    # Learning rate configurations
    use_lr_schedule: bool = True
    lr_schedule_type: Literal["cosine", "constant", "step", "linear_decay"] = "cosine"
    warmup_steps_ratio: float = 0.05  # Fraction of total steps
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of initial LR
    
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
            
            # Keep other settings from base
            gradient_accumulation_steps=base_config.gradient_accumulation_steps,
            muon_lr=base_config.muon_lr,
            muon_momentum=base_config.muon_momentum,
            adamw_lr=base_config.adamw_lr,
            max_seq_len=base_config.max_seq_len,
            num_documents=base_config.num_documents,
            max_tokens=base_config.max_tokens,
            eval_steps=base_config.eval_steps,
            use_amp=base_config.use_amp,
            vocab_size=base_config.vocab_size,
            log_milestones=base_config.log_milestones,
        )
        return config


# Define experiments to test different hypotheses
EXPERIMENTS = {
    "baseline": ExperimentConfig(
        name="baseline",
        description="Baseline: Current setup with cosine LR schedule",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "early_stopping": ExperimentConfig(
        name="early_stopping",
        description="Test early stopping to prevent overfitting",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        load_balancing_weight=0.01,
        use_early_stopping=True,
        early_stopping_patience=30,
        early_stopping_min_delta=0.001,
    ),
    
    "constant_lr": ExperimentConfig(
        name="constant_lr",
        description="Test with constant learning rate (no schedule)",
        max_steps=1000,
        use_lr_schedule=False,
        lr_schedule_type="constant",
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "lower_lb_weight": ExperimentConfig(
        name="lower_lb_weight",
        description="Test with lower load balancing weight (0.001 vs 0.01)",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        load_balancing_weight=0.001,
        use_early_stopping=False,
    ),
    
    "no_lb": ExperimentConfig(
        name="no_lb",
        description="Test without load balancing loss",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        load_balancing_weight=0.0,
        use_early_stopping=False,
    ),
    
    "higher_dropout": ExperimentConfig(
        name="higher_dropout",
        description="Test with higher dropout (0.2 vs 0.1) to reduce overfitting",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        load_balancing_weight=0.01,
        dropout=0.2,
        use_early_stopping=False,
    ),
    
    "linear_decay": ExperimentConfig(
        name="linear_decay",
        description="Test with linear LR decay instead of cosine",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="linear_decay",
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "slower_min_lr": ExperimentConfig(
        name="slower_min_lr",
        description="Test with higher minimum LR (0.3 vs 0.1) to prevent over-optimization",
        max_steps=1000,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        min_lr_ratio=0.3,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "short_run": ExperimentConfig(
        name="short_run",
        description="Shorter run to see if issue appears (600 steps)",
        max_steps=600,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
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
        print(f"  - Steps: {config.max_steps}")
        print(f"  - LR schedule: {config.lr_schedule_type if config.use_lr_schedule else 'none'}")
        print(f"  - Load balancing weight: {config.load_balancing_weight}")
        print(f"  - Early stopping: {config.use_early_stopping}")
        if config.use_early_stopping:
            print(f"    Patience: {config.early_stopping_patience} evals")
        print(f"  - Dropout: {config.dropout}")

