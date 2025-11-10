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
    max_steps: int = 500
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
    
    # Muon-specific hyperparameters
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5  # Newton-Schulz iteration steps
    
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
        description="Baseline: Hybrid Muon (2D weights) + AdamW (embeddings/norms) - LR 0.07 (optimized)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        muon_lr=0.07,
        adamw_lr=0.007,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "muon_optimal": ExperimentConfig(
        name="muon_optimal",
        description="Optimal Muon: LR=0.07, momentum=0.9 (best discovered settings)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.9,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    # Test if we can push LR higher with optimal momentum
    "muon_lr_0.09_momentum_0.9": ExperimentConfig(
        name="muon_lr_0.09_momentum_0.9",
        description="Test higher LR (0.09) with optimal momentum (0.9)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.09,
        adamw_lr=0.009,
        muon_momentum=0.9,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.1_momentum_0.9": ExperimentConfig(
        name="muon_lr_0.1_momentum_0.9",
        description="Test very high LR (0.1) with optimal momentum (0.9)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.1,
        adamw_lr=0.01,
        muon_momentum=0.9,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # Fine-tune momentum around 0.9
    "muon_momentum_0.85": ExperimentConfig(
        name="muon_momentum_0.85",
        description="Lower momentum (0.85) with optimal LR (0.07)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.85,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_momentum_0.92": ExperimentConfig(
        name="muon_momentum_0.92",
        description="Higher momentum (0.92) with optimal LR (0.07)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.92,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # Test different NS steps with optimal settings
    "muon_optimal_ns3": ExperimentConfig(
        name="muon_optimal_ns3",
        description="Optimal settings (LR=0.07, mom=0.9) with fewer NS steps (3)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.9,
        muon_ns_steps=3,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_optimal_ns10": ExperimentConfig(
        name="muon_optimal_ns10",
        description="Optimal settings (LR=0.07, mom=0.9) with more NS steps (10)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.9,
        muon_ns_steps=10,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # Test without Nesterov momentum
    "muon_optimal_no_nesterov": ExperimentConfig(
        name="muon_optimal_no_nesterov",
        description="Optimal settings (LR=0.07, mom=0.9) without Nesterov momentum",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.9,
        muon_nesterov=False,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # Test weight decay variations
    "muon_optimal_wd_0.05": ExperimentConfig(
        name="muon_optimal_wd_0.05",
        description="Optimal settings with lower weight decay (0.05)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.9,
        weight_decay=0.05,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_optimal_wd_0.2": ExperimentConfig(
        name="muon_optimal_wd_0.2",
        description="Optimal settings with higher weight decay (0.2)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.9,
        weight_decay=0.2,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # Long training with optimal settings
    "muon_optimal_long": ExperimentConfig(
        name="muon_optimal_long",
        description="Optimal settings (LR=0.07, mom=0.9) with long training (1000 steps)",
        optimizer_type="muon_hybrid",
        max_steps=1000,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.9,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=20,
    ),
    
    "adam_baseline": ExperimentConfig(
        name="adam_baseline",
        description="Pure Adam: AdamW for all parameters - LR 0.001 (optimal)",
        optimizer_type="adam",
        max_steps=500,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        adam_lr=0.001,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    # Adam experiments with optimal LR
    "adam_optimal": ExperimentConfig(
        name="adam_optimal",
        description="Optimal Adam: LR=0.001 (best discovered settings)",
        optimizer_type="adam",
        max_steps=500,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        adam_lr=0.001,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "adam_optimal_wd_0.05": ExperimentConfig(
        name="adam_optimal_wd_0.05",
        description="Adam optimal with lower weight decay (0.05)",
        optimizer_type="adam",
        max_steps=500,
        adam_lr=0.001,
        weight_decay=0.05,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "adam_optimal_wd_0.2": ExperimentConfig(
        name="adam_optimal_wd_0.2",
        description="Adam optimal with higher weight decay (0.2)",
        optimizer_type="adam",
        max_steps=500,
        adam_lr=0.001,
        weight_decay=0.2,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "adam_no_warmup": ExperimentConfig(
        name="adam_no_warmup",
        description="Adam optimal with no warmup",
        optimizer_type="adam",
        max_steps=500,
        adam_lr=0.001,
        warmup_steps_ratio=0.0,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "adam_warmup_0.1": ExperimentConfig(
        name="adam_warmup_0.1",
        description="Adam optimal with longer warmup (10%)",
        optimizer_type="adam",
        max_steps=500,
        adam_lr=0.001,
        warmup_steps_ratio=0.1,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "adam_linear_decay": ExperimentConfig(
        name="adam_linear_decay",
        description="Adam optimal with linear decay schedule",
        optimizer_type="adam",
        max_steps=500,
        adam_lr=0.001,
        use_lr_schedule=True,
        lr_schedule_type="linear_decay",
    ),
    
    "adam_constant_lr_optimal": ExperimentConfig(
        name="adam_constant_lr_optimal",
        description="Adam with constant LR (no schedule)",
        optimizer_type="adam",
        max_steps=500,
        adam_lr=0.001,
        use_lr_schedule=False,
        lr_schedule_type="constant",
    ),
    
    "adam_higher_lr": ExperimentConfig(
        name="adam_higher_lr",
        description="Adam with higher learning rate (0.002)",
        optimizer_type="adam",
        max_steps=500,
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
        max_steps=500,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        adam_lr=0.0005,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "muon_only": ExperimentConfig(
        name="muon_only",
        description="Pure Muon: Muon for all 2D parameters, minimal AdamW - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        muon_lr=0.07,
        adamw_lr=0.007,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "muon_constant_lr": ExperimentConfig(
        name="muon_constant_lr",
        description="Muon with constant learning rate (no schedule) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        use_lr_schedule=False,
        lr_schedule_type="constant",
        muon_lr=0.07,
        adamw_lr=0.007,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    "adam_constant_lr": ExperimentConfig(
        name="adam_constant_lr",
        description="Adam with constant learning rate (no schedule)",
        optimizer_type="adam",
        max_steps=500,
        use_lr_schedule=False,
        lr_schedule_type="constant",
        adam_lr=0.001,
        load_balancing_weight=0.01,
        use_early_stopping=False,
    ),
    
    # ============================================================================
    # ADAM LEARNING RATE SWEEP
    # ============================================================================
    
    # Fast 200-step LR sweep for Adam
    "adam_lr_0.0001_fast": ExperimentConfig(
        name="adam_lr_0.0001_fast",
        description="[FAST] Adam LR 0.0001 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.0001,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.0002_fast": ExperimentConfig(
        name="adam_lr_0.0002_fast",
        description="[FAST] Adam LR 0.0002 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.0002,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.0003_fast": ExperimentConfig(
        name="adam_lr_0.0003_fast",
        description="[FAST] Adam LR 0.0003 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.0003,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.0005_fast": ExperimentConfig(
        name="adam_lr_0.0005_fast",
        description="[FAST] Adam LR 0.0005 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.0005,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.0007_fast": ExperimentConfig(
        name="adam_lr_0.0007_fast",
        description="[FAST] Adam LR 0.0007 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.0007,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.001_fast": ExperimentConfig(
        name="adam_lr_0.001_fast",
        description="[FAST] Adam LR 0.001 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.001,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.002_fast": ExperimentConfig(
        name="adam_lr_0.002_fast",
        description="[FAST] Adam LR 0.002 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.002,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.003_fast": ExperimentConfig(
        name="adam_lr_0.003_fast",
        description="[FAST] Adam LR 0.003 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.003,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.005_fast": ExperimentConfig(
        name="adam_lr_0.005_fast",
        description="[FAST] Adam LR 0.005 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.005,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.007_fast": ExperimentConfig(
        name="adam_lr_0.007_fast",
        description="[FAST] Adam LR 0.007 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.007,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "adam_lr_0.01_fast": ExperimentConfig(
        name="adam_lr_0.01_fast",
        description="[FAST] Adam LR 0.01 (200 steps)",
        optimizer_type="adam",
        max_steps=200,
        adam_lr=0.01,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    # ============================================================================
    # MUON-SPECIFIC EXPERIMENTS
    # ============================================================================
    
    # 1. Learning Rate Sweep for Muon
    "muon_lr_0.005": ExperimentConfig(
        name="muon_lr_0.005",
        description="Muon with lower LR (0.005)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.005,
        adamw_lr=0.0005,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.015": ExperimentConfig(
        name="muon_lr_0.015",
        description="Muon with higher LR (0.015)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.015,
        adamw_lr=0.0015,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.02": ExperimentConfig(
        name="muon_lr_0.02",
        description="Muon with high LR (0.02)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.02,
        adamw_lr=0.002,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.03": ExperimentConfig(
        name="muon_lr_0.03",
        description="Muon with very high LR (0.03)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.03,
        adamw_lr=0.003,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.04": ExperimentConfig(
        name="muon_lr_0.04",
        description="Muon with very high LR (0.04)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.04,
        adamw_lr=0.004,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.05": ExperimentConfig(
        name="muon_lr_0.05",
        description="Muon with very high LR (0.05)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.05,
        adamw_lr=0.005,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.06": ExperimentConfig(
        name="muon_lr_0.06",
        description="Muon with extremely high LR (0.06)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.06,
        adamw_lr=0.006,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.07": ExperimentConfig(
        name="muon_lr_0.07",
        description="Muon with extremely high LR (0.07)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.08": ExperimentConfig(
        name="muon_lr_0.08",
        description="Muon with extremely high LR (0.08)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.08,
        adamw_lr=0.008,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_lr_0.1": ExperimentConfig(
        name="muon_lr_0.1",
        description="Muon with ultra high LR (0.1)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.1,
        adamw_lr=0.01,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # Fast 200-step LR sweep for quick iteration
    "muon_lr_0.02_fast": ExperimentConfig(
        name="muon_lr_0.02_fast",
        description="[FAST] Muon LR 0.02 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.02,
        adamw_lr=0.002,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.03_fast": ExperimentConfig(
        name="muon_lr_0.03_fast",
        description="[FAST] Muon LR 0.03 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.03,
        adamw_lr=0.003,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.04_fast": ExperimentConfig(
        name="muon_lr_0.04_fast",
        description="[FAST] Muon LR 0.04 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.04,
        adamw_lr=0.004,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.05_fast": ExperimentConfig(
        name="muon_lr_0.05_fast",
        description="[FAST] Muon LR 0.05 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.05,
        adamw_lr=0.005,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.06_fast": ExperimentConfig(
        name="muon_lr_0.06_fast",
        description="[FAST] Muon LR 0.06 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.06,
        adamw_lr=0.006,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.07_fast": ExperimentConfig(
        name="muon_lr_0.07_fast",
        description="[FAST] Muon LR 0.07 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.07,
        adamw_lr=0.007,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.08_fast": ExperimentConfig(
        name="muon_lr_0.08_fast",
        description="[FAST] Muon LR 0.08 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.08,
        adamw_lr=0.008,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.1_fast": ExperimentConfig(
        name="muon_lr_0.1_fast",
        description="[FAST] Muon LR 0.1 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.1,
        adamw_lr=0.01,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.12_fast": ExperimentConfig(
        name="muon_lr_0.12_fast",
        description="[FAST] Muon LR 0.12 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.12,
        adamw_lr=0.012,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    "muon_lr_0.15_fast": ExperimentConfig(
        name="muon_lr_0.15_fast",
        description="[FAST] Muon LR 0.15 (200 steps)",
        optimizer_type="muon_hybrid",
        max_steps=200,
        muon_lr=0.15,
        adamw_lr=0.015,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=10,
    ),
    
    # 2. Momentum Variations (with optimized LR=0.07)
    "muon_momentum_0.9": ExperimentConfig(
        name="muon_momentum_0.9",
        description="Muon with lower momentum (0.9) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.9,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_momentum_0.97": ExperimentConfig(
        name="muon_momentum_0.97",
        description="Muon with higher momentum (0.97) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.97,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_momentum_0.99": ExperimentConfig(
        name="muon_momentum_0.99",
        description="Muon with very high momentum (0.99) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_momentum=0.99,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # 3. Newton-Schulz Iteration Steps (with optimized LR=0.07)
    "muon_ns_steps_3": ExperimentConfig(
        name="muon_ns_steps_3",
        description="Muon with fewer NS iterations (3 steps) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_ns_steps=3,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_ns_steps_7": ExperimentConfig(
        name="muon_ns_steps_7",
        description="Muon with more NS iterations (7 steps) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_ns_steps=7,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_ns_steps_10": ExperimentConfig(
        name="muon_ns_steps_10",
        description="Muon with many NS iterations (10 steps) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_ns_steps=10,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # 4. Nesterov Momentum On/Off (with optimized LR=0.07)
    "muon_no_nesterov": ExperimentConfig(
        name="muon_no_nesterov",
        description="Muon without Nesterov momentum - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        muon_nesterov=False,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # 5. Warmup Variations (with optimized LR=0.07)
    "muon_no_warmup": ExperimentConfig(
        name="muon_no_warmup",
        description="Muon with no warmup - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        warmup_steps_ratio=0.0,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_warmup_0.1": ExperimentConfig(
        name="muon_warmup_0.1",
        description="Muon with longer warmup (10% of steps) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        warmup_steps_ratio=0.1,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_warmup_0.2": ExperimentConfig(
        name="muon_warmup_0.2",
        description="Muon with very long warmup (20% of steps) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        warmup_steps_ratio=0.2,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # 6. LR Schedule Comparisons (with optimized LR=0.07)
    "muon_linear_decay": ExperimentConfig(
        name="muon_linear_decay",
        description="Muon with linear decay schedule - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        use_lr_schedule=True,
        lr_schedule_type="linear_decay",
    ),
    
    "muon_step_decay": ExperimentConfig(
        name="muon_step_decay",
        description="Muon with step decay schedule - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.07,
        adamw_lr=0.007,
        use_lr_schedule=True,
        lr_schedule_type="step",
    ),
    
    # 7. Hybrid Ratio Experiments (Muon vs AdamW LR)
    "muon_adamw_ratio_20": ExperimentConfig(
        name="muon_adamw_ratio_20",
        description="Muon with 20x higher LR than AdamW",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.02,
        adamw_lr=0.001,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_adamw_ratio_5": ExperimentConfig(
        name="muon_adamw_ratio_5",
        description="Muon with 5x higher LR than AdamW",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.005,
        adamw_lr=0.001,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_adamw_equal_lr": ExperimentConfig(
        name="muon_adamw_equal_lr",
        description="Muon with equal LR to AdamW",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.001,
        adamw_lr=0.001,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # 8. Gradient Clipping Variations
    "muon_grad_clip_0.5": ExperimentConfig(
        name="muon_grad_clip_0.5",
        description="Muon with aggressive gradient clipping (0.5)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.01,
        adamw_lr=0.001,
        grad_clip=0.5,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_grad_clip_2.0": ExperimentConfig(
        name="muon_grad_clip_2.0",
        description="Muon with relaxed gradient clipping (2.0)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.01,
        adamw_lr=0.001,
        grad_clip=2.0,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_no_grad_clip": ExperimentConfig(
        name="muon_no_grad_clip",
        description="Muon without gradient clipping",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.01,
        adamw_lr=0.001,
        grad_clip=1e6,  # Effectively no clipping
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # 9. Combined Best Settings
    "muon_aggressive": ExperimentConfig(
        name="muon_aggressive",
        description="Muon with aggressive settings (high LR, high momentum, long warmup)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.03,
        adamw_lr=0.003,
        muon_momentum=0.99,
        warmup_steps_ratio=0.1,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_conservative": ExperimentConfig(
        name="muon_conservative",
        description="Muon with conservative settings (low LR, low momentum, short warmup)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.005,
        adamw_lr=0.0005,
        muon_momentum=0.9,
        warmup_steps_ratio=0.02,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    # 10. Longer Training (with optimized LR=0.07)
    "muon_long_training": ExperimentConfig(
        name="muon_long_training",
        description="Muon with longer training (1000 steps) - LR 0.07",
        optimizer_type="muon_hybrid",
        max_steps=1000,
        muon_lr=0.07,
        adamw_lr=0.007,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        eval_every=20,
    ),
    
    # 11. Minimum LR Variations
    "muon_min_lr_0.01": ExperimentConfig(
        name="muon_min_lr_0.01",
        description="Muon with higher minimum LR (1% of peak)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.01,
        adamw_lr=0.001,
        min_lr_ratio=0.01,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
    ),
    
    "muon_min_lr_0.5": ExperimentConfig(
        name="muon_min_lr_0.5",
        description="Muon with high minimum LR (50% of peak)",
        optimizer_type="muon_hybrid",
        max_steps=500,
        muon_lr=0.01,
        adamw_lr=0.001,
        min_lr_ratio=0.5,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
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
            print(f"  - Muon momentum: {config.muon_momentum}, Nesterov: {config.muon_nesterov}, NS steps: {config.muon_ns_steps}")
        print(f"  - LR schedule: {config.lr_schedule_type if config.use_lr_schedule else 'none'}")
        print(f"  - Warmup ratio: {config.warmup_steps_ratio}, Min LR ratio: {config.min_lr_ratio}")
        print(f"  - Grad clip: {config.grad_clip}")
        print(f"  - Load balancing weight: {config.load_balancing_weight}")
        print(f"  - Early stopping: {config.use_early_stopping}")

