"""
Experiment-specific optimizer and scheduler setup for Muon vs Adam comparison.
This module only contains experiment-specific configurations.
The actual training loop is in the base training/trainer.py module.
"""
import torch
import torch.nn as nn
import math
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths
script_dir = Path(__file__).resolve().parent.parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from configs.moe_config import MoEModelConfig
from exp_configs.experiment_configs import ExperimentConfig
from models.moe_llm import MoEMinimalLLM
from optimizers.muon import Muon
from utils.helpers import set_seed

# Import the base training infrastructure
from training.trainer import train_model, EarlyStopping


def get_lr_scheduler(optimizer, config: MoEModelConfig, exp_config: ExperimentConfig):
    """Create learning rate scheduler based on experiment config"""
    if not exp_config.use_lr_schedule or exp_config.lr_schedule_type == "constant":
        # Constant LR - no scheduling
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    
    warmup_steps = int(config.max_steps * exp_config.warmup_steps_ratio)
    min_lr_ratio = exp_config.min_lr_ratio
    
    if exp_config.lr_schedule_type == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    elif exp_config.lr_schedule_type == "linear_decay":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress)
    
    elif exp_config.lr_schedule_type == "step":
        # Step decay: reduce by 0.5 every 25% of training
        step_size = config.max_steps // 4
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 ** ((step - warmup_steps) // step_size)
    
    else:
        raise ValueError(f"Unknown lr_schedule_type: {exp_config.lr_schedule_type}")
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_muon_optimizer(model: nn.Module, config: MoEModelConfig, exp_config: ExperimentConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(
        muon_params, 
        lr=exp_config.muon_lr, 
        momentum=exp_config.muon_momentum,
        nesterov=exp_config.muon_nesterov,
        ns_steps=exp_config.muon_ns_steps
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_params, 
        lr=exp_config.adamw_lr, 
        weight_decay=config.weight_decay
    )
    
    print(f"  Muon config: LR={exp_config.muon_lr}, momentum={exp_config.muon_momentum}, "
          f"nesterov={exp_config.muon_nesterov}, ns_steps={exp_config.muon_ns_steps}")

    return [muon_optimizer, adamw_optimizer]


def setup_adam_optimizer(model: nn.Module, config: MoEModelConfig, exp_config: ExperimentConfig):
    """Setup pure Adam/AdamW optimizer for all parameters"""
    print(f"  AdamW parameters (all): {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=exp_config.adam_lr, 
        weight_decay=config.weight_decay
    )
    
    return [optimizer]


def plot_experiment_metrics(metrics_history, exp_config: ExperimentConfig, output_path: Path):
    """Plot and save experiment metrics with experiment-specific formatting"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Experiment: {exp_config.name} ({exp_config.optimizer_type})', fontsize=14, fontweight='bold')
    
    # Plot 1: Val Loss vs Time
    ax = axes[0, 0]
    ax.plot(metrics_history['elapsed_times'], metrics_history['val_losses'], 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Time')
    ax.grid(True, alpha=0.3)
    
    # Highlight best point
    best_idx = metrics_history['val_losses'].index(min(metrics_history['val_losses']))
    ax.plot(metrics_history['elapsed_times'][best_idx], 
            metrics_history['val_losses'][best_idx], 
            'r*', markersize=15, label=f'Best: {metrics_history["val_losses"][best_idx]:.4f}')
    ax.legend()
    
    # Plot 2: Val Loss vs Steps
    ax = axes[0, 1]
    ax.plot(metrics_history['steps'], metrics_history['val_losses'], 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Steps')
    ax.grid(True, alpha=0.3)
    ax.plot(metrics_history['steps'][best_idx], 
            metrics_history['val_losses'][best_idx], 
            'r*', markersize=15)
    
    # Plot 3: Val Accuracy vs Steps
    ax = axes[1, 0]
    ax.plot(metrics_history['steps'], metrics_history['val_accuracies'], 'purple', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy vs Steps')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate vs Steps
    ax = axes[1, 1]
    ax.plot(metrics_history['steps'], metrics_history['learning_rates'], 'orange', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / "metrics_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 Plots saved to {plot_path}")


def train_experiment(
    config: MoEModelConfig,
    exp_config: ExperimentConfig,
    train_loader,
    val_loader,
    output_dir: str = "./experiments"
):
    """
    Train model with experiment configuration.
    This function orchestrates experiment-specific setup and calls the base trainer.
    """
    
    # Create output directory
    output_path = Path(output_dir) / exp_config.name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🧪 Running Experiment: {exp_config.name}")
    print(f"{'='*80}")
    print(f"Description: {exp_config.description}")
    print(f"Optimizer: {exp_config.optimizer_type}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    # Initialize model
    set_seed(42)
    model = MoEMinimalLLM(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'expert' not in n)
    expert_params = total_params - active_params

    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,}")
    print(f"  📊 Expert parameters: {expert_params:,}")
    print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")

    # Setup optimizers based on experiment config
    if exp_config.optimizer_type in ["muon", "muon_hybrid"]:
        optimizers = setup_muon_optimizer(model, config, exp_config)
        print(f"  ✅ Using Muon (hybrid) optimizer")
    elif exp_config.optimizer_type == "adam":
        optimizers = setup_adam_optimizer(model, config, exp_config)
        print(f"  ✅ Using Adam optimizer")
    else:
        raise ValueError(f"Unknown optimizer_type: {exp_config.optimizer_type}")

    # Setup learning rate schedulers
    schedulers = [get_lr_scheduler(opt, config, exp_config) for opt in optimizers]
    
    # Early stopping
    early_stopper = None
    if exp_config.use_early_stopping:
        early_stopper = EarlyStopping(
            patience=exp_config.early_stopping_patience,
            min_delta=exp_config.early_stopping_min_delta
        )
        print(f"  🛑 Early stopping enabled: patience={exp_config.early_stopping_patience}")

    # Prepare experiment config for saving
    extra_config = {
        'name': exp_config.name,
        'description': exp_config.description,
        'optimizer_type': exp_config.optimizer_type,
        'max_steps': exp_config.max_steps,
        'lr_schedule_type': exp_config.lr_schedule_type,
        'use_early_stopping': exp_config.use_early_stopping,
        'load_balancing_weight': exp_config.load_balancing_weight,
        'dropout': exp_config.dropout,
        'warmup_steps_ratio': exp_config.warmup_steps_ratio,
        'min_lr_ratio': exp_config.min_lr_ratio,
        'grad_clip': exp_config.grad_clip,
    }
    
    # Add optimizer-specific config
    if exp_config.optimizer_type in ["muon", "muon_hybrid"]:
        extra_config.update({
            'muon_lr': exp_config.muon_lr,
            'adamw_lr': exp_config.adamw_lr,
            'muon_momentum': exp_config.muon_momentum,
            'muon_nesterov': exp_config.muon_nesterov,
            'muon_ns_steps': exp_config.muon_ns_steps,
        })
    elif exp_config.optimizer_type == "adam":
        extra_config.update({
            'adam_lr': exp_config.adam_lr,
        })

    # Custom plotting function that includes experiment info
    def plot_fn(metrics_history, output_path):
        plot_experiment_metrics(metrics_history, exp_config, output_path)

    # Use the base training function from training/trainer.py
    model, final_eval, metrics_history = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        early_stopper=early_stopper,
        output_dir=str(output_path),
        experiment_name=exp_config.name,
        plot_fn=plot_fn,
        extra_config=extra_config,
    )
    
    return model, final_eval, metrics_history
