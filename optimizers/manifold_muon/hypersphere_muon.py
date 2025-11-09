"""
Hypersphere Muon Optimizer

Implements manifold optimization for parameters constrained to the unit hypersphere.
The hypersphere constraint is ||w||_2 = 1 for vectors.

This optimizer is particularly useful for:
- Embedding vectors
- Router gate weights (per-row constraint)
- Any vector parameters that benefit from normalization

Key features:
- Maintains unit norm throughout training
- Simple and efficient (no dual ascent needed)
- Can be applied per-vector or per-row
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Literal


class HypersphereMuon(Optimizer):
    """
    Manifold optimizer for the unit hypersphere.
    
    Constrains parameters to have unit Euclidean norm: ||w||_2 = 1
    
    For matrix parameters, the constraint can be applied:
    - 'per_vector': Each column has unit norm (for embedding tables)
    - 'per_row': Each row has unit norm (for router gates)
    - 'whole': Entire matrix has unit Frobenius norm (rarely used)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        mode: How to apply constraint for matrices (default: 'per_vector')
            - 'per_vector': Normalize each column independently
            - 'per_row': Normalize each row independently
            - 'whole': Normalize entire matrix (global norm)
            
    Example:
        >>> # For embedding table (normalize each embedding)
        >>> embedding_params = [model.token_embedding.weight]
        >>> optimizer = HypersphereMuon(embedding_params, lr=0.02, mode='per_vector')
        >>> 
        >>> # For router gate (normalize each output dimension)
        >>> router_params = [router.gate.weight]
        >>> optimizer = HypersphereMuon(router_params, lr=0.02, mode='per_row')
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        mode: Literal['per_vector', 'per_row', 'whole'] = 'per_vector',
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if mode not in ['per_vector', 'per_row', 'whole']:
            raise ValueError(f"Invalid mode: {mode}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            mode=mode,
        )
        super().__init__(params, defaults)
        
        # Initialize parameters on manifold
        self._initialize_on_manifold()
    
    def _initialize_on_manifold(self):
        """Initialize all parameters to have unit norm."""
        for group in self.param_groups:
            mode = group['mode']
            
            for p in group['params']:
                with torch.no_grad():
                    if p.dim() == 1:
                        # Vector: simple normalization
                        p.data = p.data / (p.data.norm() + 1e-8)
                    
                    elif p.dim() == 2:
                        # Matrix: normalize according to mode
                        if mode == 'per_vector':
                            # Normalize each column
                            norms = p.data.norm(dim=0, keepdim=True)
                            p.data = p.data / (norms + 1e-8)
                        
                        elif mode == 'per_row':
                            # Normalize each row
                            norms = p.data.norm(dim=1, keepdim=True)
                            p.data = p.data / (norms + 1e-8)
                        
                        elif mode == 'whole':
                            # Normalize entire matrix
                            p.data = p.data / (p.data.norm() + 1e-8)
                    
                    else:
                        # Higher dimensional tensors: treat as batch of vectors
                        # Normalize over last dimension
                        norms = p.data.norm(dim=-1, keepdim=True)
                        p.data = p.data / (norms + 1e-8)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Algorithm for hypersphere (for simplicity, shown for single vector):
        1. Project gradient to tangent space: g_tan = g - w(w^T g)
        2. Normalize update: a = -lr * g_tan / ||g_tan||
        3. Update: w <- w + a
        4. Retract to manifold: w <- w / sqrt(1 + lr^2)
        
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
            mode = group['mode']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                w = p.data
                g = p.grad.data
                
                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                state['step'] += 1
                buf = state['momentum_buffer']
                
                # Apply momentum
                buf.mul_(momentum).add_(g, alpha=1 - momentum)
                
                if nesterov:
                    g_update = g.add(buf, alpha=momentum)
                else:
                    g_update = buf
                
                # Project to tangent space and update based on parameter shape
                if p.dim() == 1:
                    # Vector case
                    w_new = self._update_vector(w, g_update, lr)
                
                elif p.dim() == 2:
                    # Matrix case
                    if mode == 'per_vector':
                        w_new = self._update_per_column(w, g_update, lr)
                    elif mode == 'per_row':
                        w_new = self._update_per_row(w, g_update, lr)
                    elif mode == 'whole':
                        w_new = self._update_vector(w.view(-1), g_update.view(-1), lr).view(w.shape)
                
                else:
                    # Higher dimensional: update per last dimension
                    original_shape = w.shape
                    w_flat = w.view(-1, w.size(-1))
                    g_flat = g_update.view(-1, g_update.size(-1))
                    w_new_flat = self._update_per_row(w_flat, g_flat, lr)
                    w_new = w_new_flat.view(original_shape)
                
                # Update parameter
                p.data.copy_(w_new)
    
        return loss
    
    def _update_vector(self, w: torch.Tensor, g: torch.Tensor, lr: float) -> torch.Tensor:
        """
        Update a single vector on the hypersphere.
        
        Args:
            w: Current vector (unit norm)
            g: Gradient
            lr: Learning rate
            
        Returns:
            Updated vector (unit norm)
        """
        # Project gradient to tangent space: g_tan = g - w(w^T g)
        w_dot_g = torch.dot(w, g)
        g_tan = g - w * w_dot_g
        
        # Compute norm of tangent gradient
        g_tan_norm = g_tan.norm()
        
        if g_tan_norm < 1e-8:
            # Gradient is radial, no tangent component
            return w
        
        # Normalized tangent direction
        direction = g_tan / g_tan_norm
        
        # Take step in tangent space
        w_new = w - lr * direction
        
        # Retract back to sphere
        # For small lr, ||w_new|| ≈ sqrt(1 + lr^2) ≈ 1 + lr^2/2
        # We can just normalize, which is exact
        w_new = w_new / w_new.norm()
        
        return w_new
    
    def _update_per_column(self, W: torch.Tensor, G: torch.Tensor, lr: float) -> torch.Tensor:
        """
        Update matrix with per-column normalization.
        
        Each column is treated as an independent vector on the hypersphere.
        """
        W_new = W.clone()
        
        for j in range(W.size(1)):
            w = W[:, j]
            g = G[:, j]
            W_new[:, j] = self._update_vector(w, g, lr)
        
        return W_new
    
    def _update_per_row(self, W: torch.Tensor, G: torch.Tensor, lr: float) -> torch.Tensor:
        """
        Update matrix with per-row normalization.
        
        Each row is treated as an independent vector on the hypersphere.
        """
        W_new = W.clone()
        
        for i in range(W.size(0)):
            w = W[i, :]
            g = G[i, :]
            W_new[i, :] = self._update_vector(w, g, lr)
        
        return W_new
    
    def check_constraints(self, tol: float = 1e-4):
        """
        Check constraint satisfaction for all parameters.
        
        Returns:
            Dictionary with constraint statistics
        """
        stats = {
            'max_violation': 0.0,
            'mean_violation': 0.0,
            'num_violations': 0,
            'total_constraints': 0,
        }
        
        violations = []
        
        for group in self.param_groups:
            mode = group['mode']
            
            for p in group['params']:
                w = p.data
                
                if p.dim() == 1:
                    # Single vector
                    error = abs(w.norm().item() - 1.0)
                    violations.append(error)
                    stats['total_constraints'] += 1
                    if error > tol:
                        stats['num_violations'] += 1
                
                elif p.dim() == 2:
                    if mode == 'per_vector':
                        # Check each column
                        norms = w.norm(dim=0)
                        errors = (norms - 1.0).abs()
                        violations.extend(errors.tolist())
                        stats['total_constraints'] += w.size(1)
                        stats['num_violations'] += (errors > tol).sum().item()
                    
                    elif mode == 'per_row':
                        # Check each row
                        norms = w.norm(dim=1)
                        errors = (norms - 1.0).abs()
                        violations.extend(errors.tolist())
                        stats['total_constraints'] += w.size(0)
                        stats['num_violations'] += (errors > tol).sum().item()
                    
                    elif mode == 'whole':
                        # Check entire matrix
                        error = abs(w.norm().item() - 1.0)
                        violations.append(error)
                        stats['total_constraints'] += 1
                        if error > tol:
                            stats['num_violations'] += 1
        
        if violations:
            stats['max_violation'] = max(violations)
            stats['mean_violation'] = sum(violations) / len(violations)
        
        return stats
    
    def log_metrics(self, prefix: str = "optimizer") -> dict:
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
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())
        
        if grad_norms:
            metrics[f'{prefix}/grad_norm_mean'] = sum(grad_norms) / len(grad_norms)
            metrics[f'{prefix}/grad_norm_max'] = max(grad_norms)
        
        return metrics




