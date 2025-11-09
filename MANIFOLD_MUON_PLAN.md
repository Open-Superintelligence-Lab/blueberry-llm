# Manifold Muon Implementation Plan for Blueberry LLM

## Executive Summary

This document outlines a comprehensive plan to implement custom manifold optimization for each component of the Blueberry LLM, based on the modular manifolds framework. Each component will receive a tailored optimizer that respects its functional role and geometric constraints.

---

## 1. Component Analysis

### Current Architecture Components

Based on your codebase analysis:

1. **Token Embeddings** (`nn.Embedding`) - Maps token IDs to vectors
2. **Attention QKV Projection** (`nn.Linear`) - Projects to Q, K, V spaces  
3. **Attention Output Projection** (`nn.Linear`) - Projects attention output back
4. **Expert Up-Projection** (`nn.Linear` in Expert) - Projects to higher dimension
5. **Expert Down-Projection** (`nn.Linear` in Expert) - Projects back to model dimension
6. **Router Gate** (`nn.Linear` in TopKRouter) - Routes tokens to experts
7. **LM Head** (tied with embeddings) - Final output projection
8. **RMSNorm Parameters** - Scale parameters only

---

## 2. Manifold & Norm Selection by Component

### 2.1 Token Embeddings (and LM Head)

**Component Role:** Maps discrete tokens to continuous vector space

**Manifold Choice:** **Hypersphere per embedding vector**
- Each embedding vector constrained to unit sphere
- Ensures consistent embedding magnitudes
- Prevents embedding norm collapse/explosion

**Norm Choice:** **Euclidean (ℓ₂) norm**
- Natural for vector space
- Matches RMSNorm preprocessing

**Learning Rate Scaling:** Base LR (no special scaling needed)

**Benefits:**
- Normalized embeddings work well with RMSNorm
- Removes one hyperparameter (embedding scale)
- Better conditioning for downstream layers

---

### 2.2 Attention QKV Projection

**Component Role:** Transforms input to query, key, value representations

**Manifold Choice:** **Stiefel Manifold**
- Constraint: `W^T W = I` (orthonormal columns if tall, rows if wide)
- Ensures bounded condition number
- Prevents attention collapse

**Norm Choice:** **Spectral norm**
- Controls maximum singular value
- Bounds maximum effect on any input vector
- Naturally pairs with Stiefel manifold

**Learning Rate Scaling:** `lr * sqrt(fan_out / fan_in)` (from modular norm theory)

**Implementation:** Full manifold Muon with dual ascent

**Benefits:**
- Stable attention scores (no extreme dot products)
- Better gradient flow
- Prevents attention entropy collapse
- Unit condition number means predictable Lipschitz constant

---

### 2.3 Attention Output Projection

**Component Role:** Projects multi-head attention output back to residual stream

**Manifold Choice:** **Stiefel Manifold**
- Same reasoning as QKV projection
- Critical for residual stream health

**Norm Choice:** **Spectral norm**

**Learning Rate Scaling:** `lr * sqrt(fan_out / fan_in)`

**Implementation:** Full manifold Muon with dual ascent

**Special Consideration:** This is the bottleneck that combines information from multiple heads - maintaining good conditioning is critical

---

### 2.4 Expert Up-Projection (MoE FFN)

**Component Role:** Expands representation to higher dimensional space for processing

**Manifold Choice:** **Stiefel Manifold** (with special handling for rectangular matrices)

**Norm Choice:** **Spectral norm**

**Learning Rate Scaling:** `lr * sqrt(d_ff / d_model)`

**Implementation:** Manifold Muon (handles m < n case from article)

**Benefits:**
- Prevents feature explosion in high-dimensional space
- Maintains balanced activation magnitudes
- Works well with SiLU activation

---

### 2.5 Expert Down-Projection (MoE FFN)

**Component Role:** Compresses processed representation back to model dimension

**Manifold Choice:** **Stiefel Manifold**

**Norm Choice:** **Spectral norm**  

**Learning Rate Scaling:** `lr * sqrt(d_model / d_ff)`

**Implementation:** Manifold Muon

**Special Consideration:** This projection should preserve the most important features learned in the expert

---

### 2.6 Router Gate

**Component Role:** Predicts expert routing probabilities

**Manifold Choice:** **Scaled Hypersphere** (per output dimension)
- Different from Stiefel - we want routing logits to be bounded but not necessarily orthogonal
- Each row of router weight matrix on its own sphere

**Norm Choice:** **Row-wise Euclidean norm**

**Learning Rate Scaling:** Base LR

**Implementation:** Custom variant - apply hyperspherical descent per row

**Benefits:**
- Prevents routing collapse (one expert taking all tokens)
- Bounded logits prevent numerical issues
- Encourages exploration during training

**Alternative Approach:** Use full Stiefel manifold if router is square or nearly square

---

### 2.7 RMSNorm Scale Parameters

**Component Role:** Learnable scale after normalization

**Manifold Choice:** **Positive orthant with ℓ₂ constraint** (or leave unconstrained)

**Recommendation:** Use standard Adam/AdamW for these
- These are 1D parameters with different dynamics
- Manifold constraints may be too restrictive
- Standard adaptive methods work well

---

## 3. Implementation Architecture

### 3.1 Modular Optimizer Structure

```
optimizers/
├── __init__.py
├── muon.py (existing)
├── manifold_muon/
│   ├── __init__.py
│   ├── core.py                    # Core manifold operations
│   ├── stiefel_muon.py           # Stiefel manifold optimizer
│   ├── hypersphere_muon.py       # Hypersphere optimizer
│   ├── composed_optimizer.py      # Combines multiple optimizers
│   └── utils.py                   # Shared utilities
```

### 3.2 Core Manifold Operations (`core.py`)

**Required Functions:**
```python
def matrix_sign_newton_schulz(G: Tensor, steps: int = 5) -> Tensor:
    """Compute matrix sign via Newton-Schulz (already have this!)"""
    
def nuclear_norm(A: Tensor) -> Tensor:
    """Compute sum of singular values"""
    
def spectral_norm(A: Tensor) -> Tensor:
    """Compute largest singular value"""
    
def compute_dual_gradient(W: Tensor, G: Tensor, Lambda: Tensor, lr: float) -> Tensor:
    """Compute H(Λ) for dual ascent"""
    
def project_to_tangent_space(A: Tensor, W: Tensor) -> Tensor:
    """Project update to tangent space: A^T W + W^T A = 0"""
```

### 3.3 Stiefel Muon Optimizer (`stiefel_muon.py`)

**Algorithm:**
```python
class StiefelMuon(Optimizer):
    """
    Manifold Muon optimizer for Stiefel manifold.
    Constraint: W^T W = I (orthonormal columns)
    Norm: Spectral norm
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, 
                 dual_steps=10, dual_lr=0.1, ns_steps=5):
        # Store hyperparameters
        
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                # 1. Get gradient G
                # 2. Initialize/retrieve dual variable Λ
                # 3. Run dual ascent: Λ ← Λ + α·H(Λ)
                # 4. Compute optimal update: A = -η·msign(G + 2W(Λ + Λ^T))
                # 5. Apply momentum (optional)
                # 6. Update weights: W ← W + A
                # 7. Retract to manifold: W ← msign(W)
```

**Key Implementation Details:**
- Store dual variable `Λ` in optimizer state
- May need to warm-start `Λ` from previous step
- Newton-Schulz iteration for matrix sign (reuse existing code!)
- Handle both tall (m > n) and wide (m < n) matrices

### 3.4 Hypersphere Muon Optimizer (`hypersphere_muon.py`)

**Algorithm:**
```python
class HypersphereMuon(Optimizer):
    """
    Manifold optimizer for hypersphere.
    Constraint: ||w||₂ = 1
    Norm: Euclidean norm
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95):
        # Store hyperparameters
        
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                # 1. Get gradient g
                # 2. Remove radial component: g_tan = g - w(w^T g)
                # 3. Normalize: a = -lr * g_tan / ||g_tan||
                # 4. Apply momentum (optional)
                # 5. Update: w ← w + a
                # 6. Retract: w ← w / sqrt(1 + lr²)
```

**Variants:**
- Per-vector mode: For embedding table, apply to each row independently
- Per-row mode: For router gate, apply to each row independently

### 3.5 Composed Optimizer (`composed_optimizer.py`)

**Purpose:** Apply different optimizers to different parameter groups

```python
class ComposedOptimizer:
    """
    Applies different optimizers to different parameter groups.
    Implements modular norm framework's learning rate scaling.
    """
    
    def __init__(self, param_groups: List[Dict]):
        """
        param_groups: [
            {
                'params': [...],
                'optimizer': 'stiefel_muon',
                'lr_scale': 1.0,
                'base_lr': 0.02,
                ...
            },
            ...
        ]
        """
        self.optimizers = []
        for group in param_groups:
            opt_type = group['optimizer']
            lr = group['base_lr'] * group.get('lr_scale', 1.0)
            
            if opt_type == 'stiefel_muon':
                opt = StiefelMuon([group['params']], lr=lr, ...)
            elif opt_type == 'hypersphere_muon':
                opt = HypersphereMuon([group['params']], lr=lr, ...)
            elif opt_type == 'adamw':
                opt = AdamW([group['params']], lr=lr, ...)
                
            self.optimizers.append(opt)
    
    def step(self):
        for opt in self.optimizers:
            opt.step()
    
    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()
```

---

## 4. Modular Norm Learning Rate Scaling

Based on the modular manifolds theory, learning rates should be scaled per layer based on depth and fan-in/fan-out ratios.

### 4.1 Scaling Factors by Position

For a model with `L` layers, layer `l` gets scaling factor:

```python
def compute_lr_scale(layer_idx: int, total_layers: int, 
                     fan_in: int, fan_out: int) -> float:
    """
    Compute learning rate scale for modular norm.
    
    Args:
        layer_idx: Current layer index (0 to L-1)
        total_layers: Total number of layers
        fan_in: Input dimension of layer
        fan_out: Output dimension of layer
    """
    # Depth scaling: Earlier layers get smaller LR
    depth_scale = (layer_idx + 1) / total_layers
    
    # Fan-out/fan-in scaling for spectral norm
    # This is sqrt(max singular value under random init)
    ratio_scale = (fan_out / fan_in) ** 0.5
    
    return depth_scale * ratio_scale
```

### 4.2 Example Scaling for Your Architecture

Assuming `d_model=512`, `d_ff=2048`, `n_layers=12`:

| Component | Layer | fan_in | fan_out | depth_scale | ratio_scale | total_scale |
|-----------|-------|--------|---------|-------------|-------------|-------------|
| QKV proj  | 0     | 512    | 1536    | 0.08        | 1.73        | 0.14        |
| QKV proj  | 6     | 512    | 1536    | 0.50        | 1.73        | 0.87        |
| QKV proj  | 11    | 512    | 1536    | 1.00        | 1.73        | 1.73        |
| Attn out  | 0     | 512    | 512     | 0.08        | 1.00        | 0.08        |
| Expert up | 0     | 512    | 2048    | 0.08        | 2.00        | 0.16        |
| Expert down| 0    | 2048   | 512     | 0.08        | 0.50        | 0.04        |

**Key Insight:** Later layers get larger effective learning rates. This prevents early layer collapse while allowing later layers to learn quickly.

---

## 5. Integration with Training Loop

### 5.1 Modified Model Definition

Add parameter tagging to identify which optimizer each parameter should use:

```python
class MoEMinimalLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing init ...
        
        # Tag parameters with their optimizer groups
        self._tag_parameters()
    
    def _tag_parameters(self):
        """Tag each parameter with its optimizer group."""
        # Embeddings
        self.token_embedding.weight.optimizer_group = 'hypersphere'
        
        # Attention layers
        for i, block in enumerate(self.transformer_blocks):
            block.attention.qkv.weight.optimizer_group = 'stiefel'
            block.attention.qkv.weight.layer_idx = i
            block.attention.w_o.weight.optimizer_group = 'stiefel'
            block.attention.w_o.weight.layer_idx = i
            
            # MoE experts
            for expert in block.feed_forward.experts:
                expert.linear1.weight.optimizer_group = 'stiefel'
                expert.linear1.weight.layer_idx = i
                expert.linear2.weight.optimizer_group = 'stiefel'
                expert.linear2.weight.layer_idx = i
            
            # Router
            block.feed_forward.router.gate.weight.optimizer_group = 'hypersphere_per_row'
            
            # Norms - use AdamW
            block.norm1.weight.optimizer_group = 'adamw'
            block.norm2.weight.optimizer_group = 'adamw'
        
        # Output norm
        self.norm.weight.optimizer_group = 'adamw'
```

### 5.2 Optimizer Construction

```python
def create_manifold_optimizer(model: nn.Module, config: dict) -> ComposedOptimizer:
    """
    Create composed optimizer with different optimizers for different components.
    """
    # Collect parameters by group
    param_groups = {
        'stiefel': [],
        'hypersphere': [],
        'hypersphere_per_row': [],
        'adamw': []
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        group = getattr(param, 'optimizer_group', 'adamw')
        layer_idx = getattr(param, 'layer_idx', None)
        
        # Compute LR scale
        if hasattr(param, 'layer_idx') and param.dim() == 2:
            fan_in, fan_out = param.shape
            lr_scale = compute_lr_scale(
                layer_idx, 
                config['n_layers'],
                fan_in, 
                fan_out
            )
        else:
            lr_scale = 1.0
        
        param_groups[group].append({
            'param': param,
            'name': name,
            'lr_scale': lr_scale
        })
    
    # Create optimizer groups
    optimizer_groups = []
    
    if param_groups['stiefel']:
        optimizer_groups.append({
            'params': [p['param'] for p in param_groups['stiefel']],
            'optimizer': 'stiefel_muon',
            'base_lr': config['base_lr'],
            'lr_scales': [p['lr_scale'] for p in param_groups['stiefel']]
        })
    
    if param_groups['hypersphere']:
        optimizer_groups.append({
            'params': [p['param'] for p in param_groups['hypersphere']],
            'optimizer': 'hypersphere_muon',
            'base_lr': config['base_lr'],
            'lr_scales': [p['lr_scale'] for p in param_groups['hypersphere']]
        })
    
    if param_groups['adamw']:
        optimizer_groups.append({
            'params': [p['param'] for p in param_groups['adamw']],
            'optimizer': 'adamw',
            'base_lr': config['base_lr'] * 10,  # Higher LR for norm params
            'lr_scales': [1.0] * len(param_groups['adamw'])
        })
    
    return ComposedOptimizer(optimizer_groups)
```

### 5.3 Training Loop Modifications

```python
# In trainer.py
def train_step(model, batch, optimizer, config):
    optimizer.zero_grad()
    
    # Forward pass
    logits, aux_loss = model(batch['input_ids'])
    
    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        batch['labels'].view(-1)
    )
    
    if aux_loss is not None:
        loss = loss + aux_loss
    
    # Backward pass
    loss.backward()
    
    # Optional: gradient clipping in tangent space
    # (different from standard grad clipping!)
    if config.get('grad_clip'):
        clip_gradients_manifold(model, config['grad_clip'])
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()
```

---

## 6. Phased Implementation Plan

### Phase 1: Foundation (Week 1-2)
**Goal:** Implement core manifold operations and single optimizer

**Tasks:**
1. ✅ Create `optimizers/manifold_muon/` directory structure
2. ✅ Implement `core.py` with matrix operations
3. ✅ Implement `stiefel_muon.py` (single most important optimizer)
4. ✅ Write unit tests for Stiefel manifold optimizer
5. ✅ Verify constraint satisfaction (W^T W ≈ I after updates)

**Success Criteria:**
- Matrix sign computation matches existing Newton-Schulz
- Dual ascent converges on toy problems
- Stiefel constraint maintained to tolerance 1e-4

### Phase 2: Expansion (Week 3)
**Goal:** Complete optimizer suite

**Tasks:**
1. ✅ Implement `hypersphere_muon.py`
2. ✅ Implement `composed_optimizer.py`
3. ✅ Add learning rate scaling logic
4. ✅ Write integration tests

**Success Criteria:**
- All optimizers work independently
- Composed optimizer correctly dispatches
- LR scaling computes correctly

### Phase 3: Integration (Week 4)
**Goal:** Integrate with existing training

**Tasks:**
1. ✅ Add parameter tagging to model classes
2. ✅ Modify training loop to use composed optimizer
3. ✅ Add logging for per-group metrics
4. ✅ Create config options for manifold vs standard optimization

**Success Criteria:**
- Model trains without errors
- Can switch between manifold and standard optimization
- Logging shows per-component updates

### Phase 4: Validation (Week 5-6)
**Goal:** Validate performance improvements

**Tasks:**
1. ✅ Run ablation study: manifold vs standard Muon vs AdamW
2. ✅ Measure training stability (gradient norms, loss variance)
3. ✅ Measure final performance (perplexity, downstream tasks)
4. ✅ Analyze constraint satisfaction over training
5. ✅ Profile computational overhead

**Success Criteria:**
- Manifold Muon matches or exceeds baseline performance
- Constraint satisfaction maintained throughout training
- Overhead < 20% of training time

### Phase 5: Optimization (Week 7-8)
**Goal:** Improve efficiency and convergence

**Tasks:**
1. ✅ Optimize dual ascent (fewer steps, warm starting)
2. ✅ Add momentum to dual variables
3. ✅ Experiment with adaptive dual step sizes
4. ✅ Profile and optimize hotspots
5. ✅ Consider mixed precision for manifold operations

**Success Criteria:**
- Training time within 10% of baseline
- Better or equal final performance
- Stable across different model sizes

---

## 7. Hyperparameter Recommendations

### 7.1 Stiefel Muon Hyperparameters

```python
stiefel_config = {
    'base_lr': 0.02,              # Base learning rate
    'momentum': 0.95,              # Momentum coefficient (optional)
    'nesterov': True,              # Use Nesterov momentum
    'dual_steps': 10,              # Dual ascent iterations
    'dual_lr': 0.1,                # Dual ascent step size
    'ns_steps': 5,                 # Newton-Schulz iterations
    'constraint_tol': 1e-4,        # Tolerance for W^T W = I
}
```

**Tuning Guide:**
- Start with `dual_steps=10`, reduce if too slow
- Increase `dual_lr` if dual objective not decreasing
- `ns_steps=5` is usually sufficient (diminishing returns after)
- Monitor constraint violation; increase `ns_steps` if needed

### 7.2 Hypersphere Muon Hyperparameters

```python
hypersphere_config = {
    'base_lr': 0.02,              # Base learning rate
    'momentum': 0.95,              # Momentum coefficient
    'mode': 'per_vector',          # 'per_vector' or 'whole_matrix'
}
```

### 7.3 Learning Rate Schedules

**Recommendation:** Use warmup + cosine decay

```python
lr_schedule = {
    'warmup_steps': 1000,          # Linear warmup
    'max_steps': 100000,           # Total training steps
    'min_lr_ratio': 0.1,           # Final LR = 0.1 * base_lr
    'schedule': 'cosine',          # or 'linear', 'constant'
}
```

**Important:** Apply schedule to base LR, then multiply by per-layer scales

---

## 8. Monitoring and Debugging

### 8.1 Key Metrics to Log

**Per Component:**
- Gradient norm (before projection)
- Update norm (in tangent space)
- Constraint violation (||W^T W - I||_F for Stiefel)
- Singular value distribution
- Effective learning rate (after scaling)

**Per Step:**
- Dual objective value (for Stiefel Muon)
- Dual ascent convergence (number of steps to tolerance)
- Retraction error (how far off manifold before retraction)

**Example Logging:**
```python
def log_manifold_metrics(model, optimizer, step):
    metrics = {}
    
    for name, param in model.named_parameters():
        if hasattr(param, 'optimizer_group'):
            group = param.optimizer_group
            
            if group == 'stiefel' and param.dim() == 2:
                # Compute W^T W - I
                W = param.data
                constraint_error = torch.norm(W.T @ W - torch.eye(W.size(1)))
                metrics[f'{name}/constraint_error'] = constraint_error.item()
                
                # Compute singular values
                U, S, V = torch.svd(W)
                metrics[f'{name}/sigma_min'] = S.min().item()
                metrics[f'{name}/sigma_max'] = S.max().item()
                metrics[f'{name}/condition_number'] = (S.max() / S.min()).item()
    
    wandb.log(metrics, step=step)
```

### 8.2 Common Issues and Solutions

**Issue 1: Dual ascent not converging**
- **Symptom:** Dual objective oscillates or diverges
- **Solution:** Reduce `dual_lr`, increase `dual_steps`

**Issue 2: Constraint violation growing**
- **Symptom:** ||W^T W - I|| > 0.01
- **Solution:** Increase `ns_steps`, check for numerical instability

**Issue 3: Training instability**
- **Symptom:** Loss spikes, NaN gradients
- **Solution:** Reduce base learning rate, add gradient clipping

**Issue 4: Slower than expected**
- **Symptom:** Training takes >20% longer
- **Solution:** Reduce `dual_steps`, use torch.compile on manifold ops

---

## 9. Expected Benefits

### 9.1 Training Stability
- ✅ No weight norm explosion/collapse
- ✅ Predictable gradient magnitudes
- ✅ Reduced need for gradient clipping
- ✅ More stable attention patterns

### 9.2 Model Quality
- ✅ Better conditioning → better gradient flow
- ✅ Implicit regularization from manifold constraints
- ✅ More balanced expert utilization (from router constraints)
- ✅ Improved generalization (lower perplexity)

### 9.3 Hyperparameter Tuning
- ✅ Fewer hyperparameters to tune (no weight decay needed)
- ✅ More predictable learning rate effects
- ✅ Easier to transfer hyperparameters across scales
- ✅ Less sensitive to initialization

### 9.4 Theoretical Benefits
- ✅ Lipschitz guarantees for robustness
- ✅ Provable convergence properties
- ✅ Cleaner theoretical analysis
- ✅ Modular composability

---

## 10. Future Directions

### 10.1 Architecture-Optimizer Co-Design
- Design attention mechanisms that naturally live on manifolds
- Explore non-Euclidean attention (hyperbolic, Grassmann)
- Co-design activations with manifold constraints

### 10.2 Advanced Manifolds
- **Grassmann manifold:** For subspace learning in attention
- **Product manifolds:** Different constraints per head
- **Scaled manifolds:** Learn the radius of constraints

### 10.3 Efficiency Improvements
- Amortized dual ascent (solve less frequently)
- Low-rank approximations for large matrices
- Mixed precision manifold operations
- Custom CUDA kernels for manifold ops

### 10.4 Applications
- Apply to other modalities (vision, audio)
- Use for fine-tuning (manifold constraints as regularization)
- Explore for model compression
- Apply to diffusion models (following EDM2)

---

## 11. References and Further Reading

1. **Modular Manifolds Blog Post** (this document is based on)
2. **Absil, Mahony & Sepulchre** - Optimization on Manifolds (textbook)
3. **Modula Project** - https://modula.systems
4. **EDM2 Paper** - Weight constraints in diffusion models
5. **Polar Express** - Efficient matrix sign computation

---

## Appendix A: Mathematical Summary

### A.1 Stiefel Manifold Optimization

**Problem:**
```
min_A  trace(G^T A)
s.t.   ||A||_spectral ≤ η
       A^T W + W^T A = 0
```

**Solution:**
```
1. Solve dual: max_Λ -η·||G + 2W(Λ + Λ^T)||_nuclear
   via gradient ascent: Λ ← Λ + α·H(Λ)
   where H(Λ) = -η·[W^T·msign(G + 2W(Λ+Λ^T)) + msign(...)^T·W]

2. Compute update: A = -η·msign(G + 2W(Λ_opt + Λ_opt^T))

3. Update weights: W ← W + A

4. Retract: W ← msign(W)
```

### A.2 Hypersphere Optimization

**Problem:**
```
min_a  a^T g
s.t.   ||a||_2 = η
       a^T w = 0
```

**Solution:**
```
1. Project to tangent space: g_tan = g - w(w^T g)
2. Normalize: a = -η · g_tan / ||g_tan||_2
3. Update: w ← w + a
4. Retract: w ← w / sqrt(1 + η²)
```

### A.3 Learning Rate Scaling

**Modular Norm Formula:**
```
||(w₁, w₂, ..., wₙ)|| = max_i (s_i · ||w_i||_i)

where s_i = Lipschitz constant of layer i

For linear layers with spectral norm:
s_i = sqrt(fan_out / fan_in) × depth_scale
```

---

## Appendix B: Code Snippets

### B.1 Matrix Sign Function

```python
@torch.compile
def matrix_sign_newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Compute matrix sign function via Newton-Schulz iteration.
    This is the msign function used in manifold Muon.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.to(torch.float32)  # Higher precision for stability
    
    # Handle rectangular matrices
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False
    
    # Normalize
    X = X / (X.norm() + 1e-7)
    
    # Newton-Schulz iteration
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if transposed:
        X = X.T
    
    return X.to(G.dtype)
```

### B.2 Dual Gradient Computation

```python
def compute_dual_gradient(W: torch.Tensor, G: torch.Tensor, 
                          Lambda: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Compute gradient of dual function: H(Λ)
    
    H(Λ) = -η·∇_Λ ||G + 2W(Λ + Λ^T)||_nuclear
         = -η·[W^T·msign(G + 2W(Λ+Λ^T)) + msign(...)^T·W]
    """
    # Compute argument to msign
    Lambda_sym = Lambda + Lambda.T
    arg = G + 2 * W @ Lambda_sym
    
    # Compute matrix sign
    M = matrix_sign_newton_schulz(arg)
    
    # Compute gradient
    H = -lr * (W.T @ M + M.T @ W)
    
    return H
```

### B.3 Tangent Space Projection

```python
def project_to_tangent_stiefel(A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Project matrix A to tangent space of Stiefel manifold at W.
    
    Tangent space condition: A^T W + W^T A = 0
    
    Projection: A_tan = A - W·sym(W^T A)
    where sym(X) = (X + X^T) / 2
    """
    WtA = W.T @ A
    WtA_sym = (WtA + WtA.T) / 2
    A_tan = A - W @ WtA_sym
    return A_tan
```

---

## Appendix C: Testing Checklist

### Unit Tests
- [ ] Matrix sign function correctness (M^2 ≈ I)
- [ ] Tangent space projection (A^T W + W^T A ≈ 0)
- [ ] Dual gradient computation (finite difference check)
- [ ] Hypersphere retraction (||w|| ≈ 1)
- [ ] Stiefel retraction (W^T W ≈ I)

### Integration Tests
- [ ] Single parameter optimization converges
- [ ] Composed optimizer dispatches correctly
- [ ] Learning rate scaling applied correctly
- [ ] Gradient accumulation works
- [ ] Mixed precision compatible

### System Tests
- [ ] Full model training completes
- [ ] Constraints maintained over training
- [ ] Performance matches or exceeds baseline
- [ ] Scales to different model sizes
- [ ] Checkpoint save/load works

---

## Conclusion

This plan provides a comprehensive roadmap for implementing manifold Muon optimization across all components of your Blueberry LLM. The modular design allows for incremental implementation and testing, while the theoretical foundation ensures principled optimization.

**Next Steps:**
1. Review this plan and adjust priorities
2. Begin Phase 1 implementation
3. Set up experiment tracking
4. Start with small-scale validation

**Questions to Consider:**
- What model size to start validation with?
- Which baseline optimizer to compare against?
- What computational budget for experiments?
- Any specific performance targets?

Good luck with the implementation! This is an exciting direction for making neural network training more principled and automatic.




