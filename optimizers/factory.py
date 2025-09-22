"""
Optimizer factory functions for creating and configuring optimizers.

This module provides utilities for setting up optimizers with the right
parameters for different model components.
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Any, Callable
from .muon import Muon, MuonWithWarmup
from configs import AdaptiveMoEModelConfig


def setup_optimizers(
    model: nn.Module, 
    config: AdaptiveMoEModelConfig,
    use_warmup: bool = True
) -> List[torch.optim.Optimizer]:
    """
    Setup optimizers with hybrid approach for different parameter types.
    
    This function separates parameters into different groups based on their
    characteristics and assigns appropriate optimizers:
    - 2D weight matrices: Muon optimizer
    - Other parameters (embeddings, biases, norms): AdamW optimizer
    
    Args:
        model: Model to optimize
        config: Model configuration
        use_warmup: Whether to use momentum warmup for Muon
        
    Returns:
        List of optimizers [muon_optimizer, adamw_optimizer]
    """
    # Separate parameters by type
    muon_params = []
    adamw_params = []
    
    muon_param_names = []
    adamw_param_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Muon is designed for 2D weight matrices
        # Exclude embeddings and normalization layers
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'embedding' not in name and
            'norm' not in name and 
            'bias' not in name):
            muon_params.append(param)
            muon_param_names.append(name)
        else:
            adamw_params.append(param)
            adamw_param_names.append(name)

    print(f"🔧 Optimizer Setup:")
    print(f"   Muon parameters: {sum(p.numel() for p in muon_params):,} ({len(muon_params)} tensors)")
    print(f"   AdamW parameters: {sum(p.numel() for p in adamw_params):,} ({len(adamw_params)} tensors)")
    
    # Create optimizers
    optimizers = []
    
    # Muon optimizer for weight matrices
    if muon_params:
        if use_warmup:
            muon_optimizer = MuonWithWarmup(
                muon_params, 
                lr=config.muon_lr, 
                momentum=0.95,
                momentum_warmup_steps=max(300, config.max_steps // 10),
                initial_momentum=0.85,
                nesterov=True,
                ns_steps=5
            )
        else:
            muon_optimizer = Muon(
                muon_params, 
                lr=config.muon_lr, 
                momentum=0.95,
                nesterov=True,
                ns_steps=5
            )
        optimizers.append(muon_optimizer)
    
    # AdamW optimizer for other parameters
    if adamw_params:
        adamw_optimizer = torch.optim.AdamW(
            adamw_params, 
            lr=config.muon_lr * 0.1,  # Lower learning rate for AdamW
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        optimizers.append(adamw_optimizer)
    
    return optimizers


def setup_parameter_groups(
    model: nn.Module,
    config: AdaptiveMoEModelConfig
) -> List[Dict[str, Any]]:
    """
    Setup parameter groups with different learning rates and weight decay.
    
    This creates parameter groups that can be used with a single optimizer
    but with different hyperparameters for different parameter types.
    
    Args:
        model: Model to optimize
        config: Model configuration
        
    Returns:
        List of parameter group dictionaries
    """
    # Define parameter groups with different settings
    param_groups = [
        {
            "name": "weight_matrices",
            "params": [],
            "lr": config.muon_lr,
            "weight_decay": 0.0,  # Muon handles this differently
        },
        {
            "name": "embeddings",
            "params": [],
            "lr": config.muon_lr * 0.1,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "norms_and_biases",
            "params": [],
            "lr": config.muon_lr * 0.1,
            "weight_decay": 0.0,  # Don't decay biases and norms
        }
    ]
    
    # Assign parameters to groups
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.ndim == 2 and 'embedding' not in name and 'norm' not in name:
            param_groups[0]["params"].append(param)
        elif 'embedding' in name:
            param_groups[1]["params"].append(param)
        else:
            param_groups[2]["params"].append(param)
    
    # Filter out empty groups
    param_groups = [group for group in param_groups if group["params"]]
    
    return param_groups


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: AdaptiveMoEModelConfig,
    scheduler_type: str = "cosine_warmup"
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Model configuration
        scheduler_type: Type of scheduler ("cosine_warmup", "linear_warmup", "constant")
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine_warmup":
        return get_cosine_warmup_scheduler(optimizer, config)
    elif scheduler_type == "linear_warmup":
        return get_linear_warmup_scheduler(optimizer, config)
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=config.max_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    config: AdaptiveMoEModelConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create cosine annealing scheduler with warmup.
    
    Args:
        optimizer: Optimizer to schedule
        config: Model configuration
        
    Returns:
        Learning rate scheduler
    """
    warmup_steps = config.max_steps // 20  # 5% warmup
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Use LambdaLR but ensure it doesn't step until optimizer has stepped
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Reset the scheduler's internal step counter to prevent premature stepping
    scheduler._step_count = 0
    
    return scheduler


def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    config: AdaptiveMoEModelConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create linear warmup followed by linear decay scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Model configuration
        
    Returns:
        Learning rate scheduler
    """
    warmup_steps = config.max_steps // 10  # 10% warmup
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Linear decay
            progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
            return 1.0 - 0.9 * progress  # Decay to 10% of original LR
    
    # Use LambdaLR but ensure it doesn't step until optimizer has stepped
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Reset the scheduler's internal step counter to prevent premature stepping
    scheduler._step_count = 0
    
    return scheduler


def get_optimizer_info(optimizers: List[torch.optim.Optimizer]) -> Dict[str, Any]:
    """
    Get information about optimizers.
    
    Args:
        optimizers: List of optimizers
        
    Returns:
        Dictionary with optimizer information
    """
    info = {
        "num_optimizers": len(optimizers),
        "optimizers": []
    }
    
    for i, opt in enumerate(optimizers):
        opt_info = {
            "index": i,
            "type": type(opt).__name__,
            "num_param_groups": len(opt.param_groups),
            "total_params": sum(
                sum(p.numel() for p in group["params"]) 
                for group in opt.param_groups
            )
        }
        
        # Add optimizer-specific info
        if hasattr(opt, 'defaults'):
            opt_info["config"] = opt.defaults.copy()
        
        info["optimizers"].append(opt_info)
    
    return info


def apply_weight_decay_exclusions(
    model: nn.Module,
    weight_decay: float,
    exclude_patterns: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with weight decay exclusions.
    
    Args:
        model: Model to optimize
        weight_decay: Weight decay value
        exclude_patterns: Patterns for parameter names to exclude from weight decay
        
    Returns:
        List of parameter groups
    """
    if exclude_patterns is None:
        exclude_patterns = ["bias", "norm", "embedding"]
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter should be excluded from weight decay
        exclude = any(pattern in name for pattern in exclude_patterns)
        
        if exclude:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    
    return param_groups


def create_single_optimizer(
    model: nn.Module,
    config: AdaptiveMoEModelConfig,
    optimizer_type: str = "adamw"
) -> torch.optim.Optimizer:
    """
    Create a single optimizer for the entire model.
    
    Args:
        model: Model to optimize
        config: Model configuration
        optimizer_type: Type of optimizer ("adamw", "muon", "sgd")
        
    Returns:
        Optimizer instance
    """
    # Create parameter groups with weight decay exclusions
    param_groups = apply_weight_decay_exclusions(model, config.weight_decay)
    
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=config.muon_lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == "muon":
        # Filter to only 2D parameters for Muon
        muon_param_groups = []
        for group in param_groups:
            muon_params = [p for p in group["params"] if p.ndim == 2]
            if muon_params:
                muon_group = group.copy()
                muon_group["params"] = muon_params
                muon_param_groups.append(muon_group)
        
        if not muon_param_groups:
            raise ValueError("No 2D parameters found for Muon optimizer")
            
        return Muon(muon_param_groups, lr=config.muon_lr, momentum=0.95)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=config.muon_lr,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
