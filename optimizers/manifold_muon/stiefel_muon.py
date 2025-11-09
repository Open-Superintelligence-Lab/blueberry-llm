"""
Stiefel Muon Optimizer

Implements manifold optimization for parameters constrained to the Stiefel manifold.
The Stiefel manifold consists of matrices W ∈ R^{m×n} with W^T W = I.

This optimizer solves the manifold Muon problem via dual ascent:
    min_{A} trace(G^T A)
    s.t.    ||A||_spectral ≤ η
            A^T W + W^T A = 0

Key features:
- Maintains unit condition number (all singular values = 1)
- Bounds spectral norm of updates
- Natural for linear layers in neural networks
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Dict, Any, Optional
from .core import (
    compute_dual_gradient,
    compute_optimal_update,
    retract_to_stiefel,
    check_stiefel_constraint,
    initialize_on_stiefel,
)


class StiefelMuon(Optimizer):
    """
    Manifold Muon optimizer for the Stiefel manifold.
    
    This optimizer maintains the constraint W^T W = I while performing
    gradient descent with spectral norm constraint on updates.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        dual_steps: Number of dual ascent iterations (default: 10)
        dual_lr: Learning rate for dual ascent (default: 0.1)
        ns_steps: Newton-Schulz iterations for matrix sign (default: 5)
        constraint_tol: Tolerance for constraint violation (default: 1e-4)
        init_on_manifold: Initialize parameters on manifold (default: True)
        
    Example:
        >>> model = MyModel()
        >>> attention_params = [p for name, p in model.named_parameters() 
        ...                     if 'attention' in name and p.dim() == 2]
        >>> optimizer = StiefelMuon(attention_params, lr=0.02)
        >>> 
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        dual_steps: int = 10,
        dual_lr: float = 0.1,
        ns_steps: int = 5,
        constraint_tol: float = 1e-4,
        init_on_manifold: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if dual_steps < 1:
            raise ValueError(f"Invalid dual_steps: {dual_steps}")
        if dual_lr <= 0.0:
            raise ValueError(f"Invalid dual_lr: {dual_lr}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            dual_steps=dual_steps,
            dual_lr=dual_lr,
            ns_steps=ns_steps,
            constraint_tol=constraint_tol,
        )
        super().__init__(params, defaults)
        
        # Initialize parameters on manifold if requested
        if init_on_manifold:
            self._initialize_on_manifold()
    
    def _initialize_on_manifold(self):
        """Initialize all parameters to lie on the Stiefel manifold."""
        for group in self.param_groups:
            for p in group['params']:
                if p.dim() != 2:
                    continue
                
                m, n = p.shape
                if m < n:
                    # Can't have orthonormal columns if m < n
                    # Transpose the constraint: W W^T = I instead
                    p.data = p.data.T
                    p.data = initialize_on_stiefel((n, m), p.device).T
                else:
                    p.data = initialize_on_stiefel((m, n), p.device)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            loss: Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            dual_steps = group['dual_steps']
            dual_lr = group['dual_lr']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Only apply to 2D parameters (matrices)
                if p.dim() != 2:
                    # Fall back to simple gradient descent for non-matrix params
                    p.add_(p.grad, alpha=-lr)
                    continue
                
                W = p.data
                G = p.grad.data
                m, n = W.shape
                
                # Handle wide matrices by transposing
                transposed = False
                if m < n:
                    W = W.T
                    G = G.T
                    m, n = n, m
                    transposed = True
                
                # Get or initialize optimizer state
                state = self.state[p]
                
                # Initialize state on first step
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(G)
                    state['dual_variable'] = torch.zeros(n, n, device=W.device, dtype=W.dtype)
                
                state['step'] += 1
                
                # Apply momentum to gradient
                buf = state['momentum_buffer']
                if transposed:
                    buf = buf.T
                
                buf.mul_(momentum).add_(G, alpha=1 - momentum)
                
                if nesterov:
                    G_update = G.add(buf, alpha=momentum)
                else:
                    G_update = buf
                
                # Dual ascent to solve for optimal Lambda
                Lambda = state['dual_variable']
                
                for dual_step in range(dual_steps):
                    # Compute dual gradient H(Lambda)
                    H = compute_dual_gradient(W, G_update, Lambda, lr, ns_steps)
                    
                    # Gradient ascent step on dual variable
                    Lambda = Lambda + dual_lr * H
                    
                    # Optional: Check for convergence
                    # if dual_step > 0 and torch.norm(H) < 1e-6:
                    #     break
                
                # Store updated dual variable (for warm starting next iteration)
                state['dual_variable'] = Lambda
                
                # Compute optimal primal update
                A_opt = compute_optimal_update(W, G_update, Lambda, lr, ns_steps)
                
                # Update weights
                W_new = W + A_opt
                
                # Retract back to manifold
                W_new = retract_to_stiefel(W_new, ns_steps)
                
                # Transpose back if needed
                if transposed:
                    W_new = W_new.T
                    buf = buf.T
                
                # Update parameter
                p.data.copy_(W_new)
                
                # Update momentum buffer
                state['momentum_buffer'] = buf if not transposed else buf.T
        
        return loss
    
    def check_constraints(self) -> Dict[str, Any]:
        """
        Check constraint satisfaction for all parameters.
        
        Returns:
            Dictionary with constraint statistics
        """
        stats = {
            'max_violation': 0.0,
            'mean_violation': 0.0,
            'num_violations': 0,
            'total_params': 0,
        }
        
        violations = []
        
        for group in self.param_groups:
            constraint_tol = group['constraint_tol']
            
            for p in group['params']:
                if p.dim() != 2:
                    continue
                
                stats['total_params'] += 1
                W = p.data
                
                # Handle wide matrices
                if W.shape[0] < W.shape[1]:
                    W = W.T
                
                satisfied, error = check_stiefel_constraint(W, constraint_tol)
                violations.append(error)
                
                if not satisfied:
                    stats['num_violations'] += 1
        
        if violations:
            stats['max_violation'] = max(violations)
            stats['mean_violation'] = sum(violations) / len(violations)
        
        return stats
    
    def log_metrics(self, prefix: str = "optimizer") -> Dict[str, float]:
        """
        Compute metrics for logging.
        
        Args:
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        constraint_stats = self.check_constraints()
        
        metrics[f'{prefix}/constraint_max_violation'] = constraint_stats['max_violation']
        metrics[f'{prefix}/constraint_mean_violation'] = constraint_stats['mean_violation']
        metrics[f'{prefix}/constraint_violations'] = constraint_stats['num_violations']
        
        # Collect gradient norms
        grad_norms = []
        update_norms = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.dim() == 2:
                    grad_norms.append(p.grad.norm().item())
                    
                    # Get update norm from last step (stored in momentum buffer)
                    state = self.state.get(p)
                    if state and 'momentum_buffer' in state:
                        update_norms.append(state['momentum_buffer'].norm().item())
        
        if grad_norms:
            metrics[f'{prefix}/grad_norm_mean'] = sum(grad_norms) / len(grad_norms)
            metrics[f'{prefix}/grad_norm_max'] = max(grad_norms)
        
        if update_norms:
            metrics[f'{prefix}/update_norm_mean'] = sum(update_norms) / len(update_norms)
            metrics[f'{prefix}/update_norm_max'] = max(update_norms)
        
        return metrics


class StiefelMuonWithSchedule(StiefelMuon):
    """
    Stiefel Muon with per-parameter learning rate scaling.
    
    This variant implements the modular norm framework's learning rate
    scaling based on layer depth and fan-in/fan-out ratios.
    
    Args:
        param_groups: List of parameter groups, each with:
            - 'params': list of parameters
            - 'lr_scale': multiplier for base learning rate
        **kwargs: Other arguments passed to StiefelMuon
        
    Example:
        >>> param_groups = [
        ...     {'params': layer1_params, 'lr_scale': 0.5},
        ...     {'params': layer2_params, 'lr_scale': 1.0},
        ...     {'params': layer3_params, 'lr_scale': 1.5},
        ... ]
        >>> optimizer = StiefelMuonWithSchedule(param_groups, lr=0.02)
    """
    
    def __init__(self, params, lr: float = 0.02, **kwargs):
        # Handle both single param list and param groups
        if isinstance(params, list) and len(params) > 0:
            if isinstance(params[0], dict):
                # Already have param groups with lr_scale
                param_groups = params
            else:
                # Single list of params, wrap in default group
                param_groups = [{'params': params, 'lr_scale': 1.0}]
        else:
            param_groups = [{'params': params, 'lr_scale': 1.0}]
        
        # Ensure each group has lr_scale
        for group in param_groups:
            if 'lr_scale' not in group:
                group['lr_scale'] = 1.0
        
        # Call parent init with modified param groups
        super().__init__(param_groups, lr=lr, **kwargs)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Step with per-parameter learning rate scaling."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Temporarily modify learning rates
        original_lrs = []
        for group in self.param_groups:
            original_lr = group['lr']
            original_lrs.append(original_lr)
            group['lr'] = original_lr * group.get('lr_scale', 1.0)
        
        # Call parent step
        super().step(closure=None)
        
        # Restore original learning rates
        for group, original_lr in zip(self.param_groups, original_lrs):
            group['lr'] = original_lr
        
        return loss




