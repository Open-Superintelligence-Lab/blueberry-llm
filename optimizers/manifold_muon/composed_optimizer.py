"""
Composed Optimizer

Combines multiple optimizers for different parameter groups.
This is the main interface for applying manifold optimization to different
parts of a neural network.

Based on the modular manifolds framework, this optimizer:
1. Groups parameters by type (Stiefel, Hypersphere, unconstrained)
2. Applies appropriate optimizer to each group
3. Handles learning rate scaling based on layer depth and dimensions
"""

import torch
from torch.optim import Optimizer, AdamW
from typing import List, Dict, Any, Optional
from .stiefel_muon import StiefelMuon, StiefelMuonWithSchedule
from .hypersphere_muon import HypersphereMuon


class ComposedOptimizer:
    """
    Composed optimizer that applies different optimizers to different parameter groups.
    
    This is the main interface for training models with manifold constraints.
    It automatically dispatches parameters to appropriate optimizers based on
    their type and applies learning rate scaling.
    
    Args:
        param_groups: List of parameter group dictionaries, each containing:
            - 'params': list of parameters
            - 'optimizer': one of 'stiefel_muon', 'hypersphere_muon', 'adamw'
            - 'lr_scale': (optional) learning rate multiplier
            - Other optimizer-specific arguments
        base_lr: Base learning rate (default: 0.02)
        **optimizer_kwargs: Default kwargs for all optimizers
        
    Example:
        >>> # Collect parameter groups from model
        >>> param_groups = [
        ...     {
        ...         'params': [attn.qkv.weight for attn in model.attentions],
        ...         'optimizer': 'stiefel_muon',
        ...         'lr_scale': 1.0,
        ...     },
        ...     {
        ...         'params': [emb.weight for emb in model.embeddings],
        ...         'optimizer': 'hypersphere_muon',
        ...         'mode': 'per_vector',
        ...     },
        ...     {
        ...         'params': [norm.weight for norm in model.norms],
        ...         'optimizer': 'adamw',
        ...         'lr_scale': 10.0,
        ...     }
        ... ]
        >>> optimizer = ComposedOptimizer(param_groups, base_lr=0.02)
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
    """
    
    def __init__(
        self,
        param_groups: List[Dict[str, Any]],
        base_lr: float = 0.02,
        **optimizer_kwargs
    ):
        self.base_lr = base_lr
        self.param_groups = param_groups
        self.optimizers = []
        
        # Create individual optimizers for each group
        for group in param_groups:
            opt_type = group.get('optimizer', 'adamw')
            params = group['params']
            lr_scale = group.get('lr_scale', 1.0)
            lr = base_lr * lr_scale
            
            # Merge group-specific kwargs with defaults
            opt_kwargs = {**optimizer_kwargs}
            opt_kwargs.update({k: v for k, v in group.items() 
                             if k not in ['params', 'optimizer', 'lr_scale']})
            
            # Create appropriate optimizer
            if opt_type == 'stiefel_muon':
                opt = StiefelMuon(params, lr=lr, **opt_kwargs)
            
            elif opt_type == 'hypersphere_muon':
                opt = HypersphereMuon(params, lr=lr, **opt_kwargs)
            
            elif opt_type == 'adamw':
                # Extract AdamW-specific params
                adamw_kwargs = {
                    'betas': opt_kwargs.get('betas', (0.9, 0.999)),
                    'eps': opt_kwargs.get('eps', 1e-8),
                    'weight_decay': opt_kwargs.get('weight_decay', 0.01),
                }
                opt = AdamW(params, lr=lr, **adamw_kwargs)
            
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")
            
            self.optimizers.append({
                'optimizer': opt,
                'type': opt_type,
                'lr_scale': lr_scale,
            })
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero out gradients for all optimizers."""
        for opt_dict in self.optimizers:
            opt_dict['optimizer'].zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        """Perform optimization step for all optimizers."""
        loss = None
        for opt_dict in self.optimizers:
            result = opt_dict['optimizer'].step(closure=closure)
            if result is not None:
                loss = result
        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict containing all optimizer states."""
        return {
            'base_lr': self.base_lr,
            'optimizers': [
                {
                    'type': opt_dict['type'],
                    'lr_scale': opt_dict['lr_scale'],
                    'state_dict': opt_dict['optimizer'].state_dict(),
                }
                for opt_dict in self.optimizers
            ]
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict for all optimizers."""
        self.base_lr = state_dict['base_lr']
        
        if len(state_dict['optimizers']) != len(self.optimizers):
            raise ValueError("State dict does not match current optimizer configuration")
        
        for opt_dict, state in zip(self.optimizers, state_dict['optimizers']):
            if opt_dict['type'] != state['type']:
                raise ValueError(f"Optimizer type mismatch: {opt_dict['type']} vs {state['type']}")
            opt_dict['optimizer'].load_state_dict(state['state_dict'])
    
    def get_lr(self) -> List[float]:
        """Get current learning rate for each optimizer."""
        lrs = []
        for opt_dict in self.optimizers:
            opt = opt_dict['optimizer']
            if hasattr(opt, 'param_groups'):
                lrs.extend([group['lr'] for group in opt.param_groups])
        return lrs
    
    def set_lr(self, lr: float):
        """Set base learning rate (scales are preserved)."""
        self.base_lr = lr
        for opt_dict in self.optimizers:
            scaled_lr = lr * opt_dict['lr_scale']
            opt = opt_dict['optimizer']
            if hasattr(opt, 'param_groups'):
                for group in opt.param_groups:
                    group['lr'] = scaled_lr
    
    def check_constraints(self) -> Dict[str, Any]:
        """Check manifold constraints for all optimizers."""
        stats = {
            'total_violations': 0,
            'by_type': {}
        }
        
        for opt_dict in self.optimizers:
            opt = opt_dict['optimizer']
            opt_type = opt_dict['type']
            
            if hasattr(opt, 'check_constraints'):
                constraint_stats = opt.check_constraints()
                stats['by_type'][opt_type] = constraint_stats
                stats['total_violations'] += constraint_stats.get('num_violations', 0)
        
        return stats
    
    def log_metrics(self, prefix: str = "optimizer") -> Dict[str, Any]:
        """
        Collect metrics from all optimizers for logging.
        
        Args:
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for i, opt_dict in enumerate(self.optimizers):
            opt = opt_dict['optimizer']
            opt_type = opt_dict['type']
            
            if hasattr(opt, 'log_metrics'):
                opt_metrics = opt.log_metrics(prefix=f"{prefix}/{opt_type}_{i}")
                metrics.update(opt_metrics)
        
        # Add constraint summary
        constraint_stats = self.check_constraints()
        metrics[f'{prefix}/total_constraint_violations'] = constraint_stats['total_violations']
        
        return metrics


def create_manifold_optimizer_from_model(
    model: torch.nn.Module,
    base_lr: float = 0.02,
    config: Optional[Dict[str, Any]] = None
) -> ComposedOptimizer:
    """
    Automatically create a composed manifold optimizer from a model.
    
    This function inspects the model and groups parameters based on their
    type and role, then creates appropriate optimizers.
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate
        config: Optional configuration dict with optimizer settings
        
    Returns:
        ComposedOptimizer configured for the model
        
    Example:
        >>> model = MyTransformer(...)
        >>> optimizer = create_manifold_optimizer_from_model(
        ...     model, 
        ...     base_lr=0.02,
        ...     config={'dual_steps': 10}
        ... )
    """
    if config is None:
        config = {}
    
    # Collect parameters by type
    stiefel_params = []
    hypersphere_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter has optimizer_group attribute
        opt_group = getattr(param, 'optimizer_group', None)
        
        if opt_group == 'stiefel':
            # Get layer-specific LR scale
            lr_scale = getattr(param, 'lr_scale', 1.0)
            stiefel_params.append({
                'param': param,
                'name': name,
                'lr_scale': lr_scale
            })
        
        elif opt_group == 'hypersphere':
            hypersphere_params.append({
                'param': param,
                'name': name,
                'lr_scale': getattr(param, 'lr_scale', 1.0)
            })
        
        else:
            # Default to AdamW for unconstrained parameters
            adamw_params.append({
                'param': param,
                'name': name,
                'lr_scale': getattr(param, 'lr_scale', 10.0)  # Higher LR for norms
            })
    
    # Create parameter groups
    param_groups = []
    
    if stiefel_params:
        param_groups.append({
            'params': [p['param'] for p in stiefel_params],
            'optimizer': 'stiefel_muon',
            'dual_steps': config.get('dual_steps', 10),
            'dual_lr': config.get('dual_lr', 0.1),
            'ns_steps': config.get('ns_steps', 5),
        })
    
    if hypersphere_params:
        param_groups.append({
            'params': [p['param'] for p in hypersphere_params],
            'optimizer': 'hypersphere_muon',
            'mode': config.get('hypersphere_mode', 'per_vector'),
        })
    
    if adamw_params:
        param_groups.append({
            'params': [p['param'] for p in adamw_params],
            'optimizer': 'adamw',
            'lr_scale': 10.0,
            'weight_decay': config.get('weight_decay', 0.01),
        })
    
    return ComposedOptimizer(param_groups, base_lr=base_lr)


def compute_lr_scale(
    layer_idx: int, 
    total_layers: int, 
    fan_in: int, 
    fan_out: int
) -> float:
    """
    Compute learning rate scale based on modular norm theory.
    
    The scale is: (layer_idx + 1) / total_layers * sqrt(fan_out / fan_in)
    
    Args:
        layer_idx: Current layer index (0 to total_layers-1)
        total_layers: Total number of layers
        fan_in: Input dimension
        fan_out: Output dimension
        
    Returns:
        Learning rate scale factor
    """
    depth_scale = (layer_idx + 1) / total_layers
    ratio_scale = (fan_out / fan_in) ** 0.5
    return depth_scale * ratio_scale




