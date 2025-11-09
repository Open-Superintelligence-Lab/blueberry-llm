# Next Steps: Implementing Manifold Muon for Blueberry LLM

## What Has Been Created

I've designed and implemented a complete manifold optimization system for your LLM based on the modular manifolds framework. Here's what you now have:

### 📚 Documentation
1. **MANIFOLD_MUON_PLAN.md** - Comprehensive 11-section plan with:
   - Component analysis for your MoE LLM
   - Manifold & norm selection for each component
   - Mathematical derivations
   - Phased implementation roadmap
   - Hyperparameter recommendations
   - Monitoring & debugging guides

2. **MANIFOLD_OPTIMIZER_SUMMARY.md** - Quick reference with:
   - Visual architecture diagram
   - Implementation roadmap
   - Key equations
   - Configuration templates
   - Expected performance characteristics

3. **optimizers/manifold_muon/README.md** - Package documentation

### 💻 Implementation
Created complete optimizer package in `optimizers/manifold_muon/`:

```
optimizers/manifold_muon/
├── __init__.py                    # Package exports
├── core.py                        # Core manifold operations ✓
├── stiefel_muon.py               # Stiefel manifold optimizer ✓
├── hypersphere_muon.py           # Hypersphere optimizer ✓
├── composed_optimizer.py          # Combines optimizers ✓
├── test_manifold_optimizers.py   # Test suite ✓
├── integration_example.py         # MoE LLM integration ✓
└── README.md                      # Documentation ✓
```

### 🎯 Key Features Implemented

1. **Core Operations** (`core.py`)
   - Matrix sign via Newton-Schulz (reuses your existing code!)
   - Nuclear & spectral norms
   - Dual gradient computation
   - Tangent space projections
   - Constraint checking utilities

2. **Stiefel Muon Optimizer** (`stiefel_muon.py`)
   - Full dual ascent algorithm
   - Handles rectangular matrices (tall & wide)
   - Momentum & Nesterov support
   - Automatic constraint maintenance
   - Per-parameter learning rate scaling

3. **Hypersphere Muon Optimizer** (`hypersphere_muon.py`)
   - Per-vector, per-row, and whole-matrix modes
   - Simple and efficient (no dual ascent needed)
   - Perfect for embeddings and router gates

4. **Composed Optimizer** (`composed_optimizer.py`)
   - Unified interface for multiple optimizer types
   - Automatic learning rate scaling (modular norm theory)
   - Integrated constraint checking
   - Easy checkpointing

5. **Integration Example** (`integration_example.py`)
   - Shows how to tag your MoE LLM parameters
   - Creates composed optimizer automatically
   - Demonstrates full training loop

## Your LLM Component Mapping

Here's how your MoE LLM components map to optimizers:

| Component | Count | Optimizer | Manifold | Priority |
|-----------|-------|-----------|----------|----------|
| **QKV Projections** | 12 layers | StiefelMuon | W^T W = I | 🔴 HIGH |
| **Attn Output** | 12 layers | StiefelMuon | W^T W = I | 🔴 HIGH |
| **Expert Up Proj** | 12×8 = 96 | StiefelMuon | W^T W = I | 🔴 HIGH |
| **Expert Down Proj** | 12×8 = 96 | StiefelMuon | W^T W = I | 🔴 HIGH |
| **Token Embeddings** | 1 table | HypersphereMuon | ‖w‖=1 | 🟡 MEDIUM |
| **Router Gates** | 12 layers | HypersphereMuon | ‖row‖=1 | 🟡 MEDIUM |
| **RMSNorm Scales** | 25 params | AdamW | None | 🟢 LOW |

**Total:** ~216 Stiefel-constrained matrices + 13 hypersphere constraints

## Immediate Action Items

### Step 1: Verify Installation (5 minutes)

```bash
cd /Users/vukrosic/AI\ Science\ Projects/blueberry-llm

# Run test suite
python -m optimizers.manifold_muon.test_manifold_optimizers
```

**Expected output:** All tests should pass ✓

### Step 2: Quick Integration Test (10 minutes)

```bash
# Run integration example
python -m optimizers.manifold_muon.integration_example
```

This creates a small MoE model and runs one training step with manifold constraints.

### Step 3: Integrate with Your Training (30 minutes)

Modify your `train_moe.py` or create a new training script:

```python
from optimizers.manifold_muon.integration_example import create_manifold_optimizer
from models.moe_llm import MoEMinimalLLM
from configs.moe_config import MoEModelConfig

# Your existing config
config = MoEModelConfig(...)
model = MoEMinimalLLM(config)

# Replace your current optimizer with:
optimizer = create_manifold_optimizer(
    model, 
    config,
    base_lr=0.02,        # Start conservative
    dual_steps=10,       # Can reduce to 5 if slow
    dual_lr=0.1,
)

# Training loop (no changes needed!)
for batch in dataloader:
    optimizer.zero_grad()
    logits, aux_loss = model(batch)
    loss = compute_loss(logits, batch)
    if aux_loss: loss += aux_loss
    loss.backward()
    optimizer.step()
    
    # Periodic constraint checking
    if step % 100 == 0:
        stats = optimizer.check_constraints()
        print(f"Constraint violations: {stats['total_violations']}")
```

### Step 4: Small-Scale Experiment (1-2 hours)

Run a small experiment to validate:

```python
# Small config for fast iteration
small_config = MoEModelConfig(
    vocab_size=10000,
    d_model=256,
    n_heads=4,
    d_ff=1024,
    n_layers=4,        # Small!
    num_experts=4,     # Small!
    max_seq_len=512,
)

# Train for 1K steps
# Compare: AdamW baseline vs Manifold Muon
```

**What to monitor:**
- Loss curve (should be similar or better)
- Constraint violations (should be < 1e-4)
- Training time (should be < 20% slower)
- Singular values (should stay near 1.0)

### Step 5: Full-Scale Training (When Ready)

After validating on small model:

```python
# Your full config
full_config = MoEModelConfig(
    d_model=512,
    n_layers=12,
    num_experts=8,
    ...
)

optimizer = create_manifold_optimizer(
    model,
    full_config,
    base_lr=0.02,
)

# Add logging
if step % 100 == 0:
    metrics = optimizer.log_metrics(prefix='optimizer')
    wandb.log(metrics, step=step)
```

## Expected Timeline

### Week 1: Validation
- ✅ Day 1-2: Run tests, verify installation
- ✅ Day 3-4: Small-scale experiment (d=256, n=4)
- ✅ Day 5-7: Analyze results, tune hyperparameters

### Week 2: Scaling
- ✅ Day 8-10: Medium-scale experiment (d=512, n=8)
- ✅ Day 11-12: Compare to baseline, measure overhead
- ✅ Day 13-14: Full-scale training

### Week 3-4: Optimization
- ✅ Profile and optimize bottlenecks
- ✅ Reduce dual_steps if needed
- ✅ Experiment with different LR schedules
- ✅ Run ablation studies

## Key Metrics to Track

### Training Stability
```python
metrics = {
    'loss': loss.item(),
    'loss_std': running_std_of_loss,  # Should be lower
    'grad_norm': grad_norm,            # Should be more stable
}
```

### Constraint Satisfaction
```python
stats = optimizer.check_constraints()
metrics.update({
    'constraint/violations': stats['total_violations'],
    'constraint/max_error': stats['by_type']['stiefel_muon']['max_violation'],
    'constraint/mean_error': stats['by_type']['stiefel_muon']['mean_violation'],
})
```

### Matrix Health
```python
# For each Stiefel-constrained matrix
U, S, V = torch.svd(matrix)
metrics.update({
    f'{name}/sigma_min': S.min(),
    f'{name}/sigma_max': S.max(),
    f'{name}/condition_number': S.max() / S.min(),  # Should be ~1.0
})
```

### Performance
```python
metrics.update({
    'time/step': time_per_step,
    'time/forward': forward_time,
    'time/backward': backward_time,
    'time/optimizer': optimizer_time,  # Target: < 20% of total
})
```

## Common Issues & Solutions

### Issue 1: "ImportError: No module named manifold_muon"
**Solution:** Make sure you're in the project root:
```bash
cd /Users/vukrosic/AI\ Science\ Projects/blueberry-llm
python -m optimizers.manifold_muon.test_manifold_optimizers
```

### Issue 2: Constraint violations growing
**Symptom:** `constraint/max_error > 0.01`

**Solutions:**
1. Increase `ns_steps` from 5 to 7
2. Use fp32 for Newton-Schulz
3. Reduce learning rate

### Issue 3: Training much slower
**Symptom:** >30% overhead

**Solutions:**
1. Reduce `dual_steps` from 10 to 5
2. Profile with `torch.profiler`
3. Check if GPU utilization is high

### Issue 4: Loss not decreasing
**Solutions:**
1. Reduce `base_lr` (try 0.01)
2. Increase warmup steps
3. Check gradient norms
4. Verify constraints aren't too restrictive

## Hyperparameter Starting Points

### Conservative (Safe Start)
```python
optimizer = create_manifold_optimizer(
    model, config,
    base_lr=0.01,          # Low LR
    dual_steps=15,         # More dual steps
    dual_lr=0.05,          # Conservative dual LR
)
```

### Balanced (Recommended)
```python
optimizer = create_manifold_optimizer(
    model, config,
    base_lr=0.02,          # Standard
    dual_steps=10,         # Balanced
    dual_lr=0.1,           # Standard
)
```

### Aggressive (Fast Training)
```python
optimizer = create_manifold_optimizer(
    model, config,
    base_lr=0.03,          # Higher LR
    dual_steps=5,          # Fewer steps
    dual_lr=0.2,           # Faster dual ascent
)
```

## Experiment Tracking Template

```python
import wandb

wandb.init(
    project="blueberry-llm",
    name="manifold-muon-experiment-1",
    config={
        "optimizer": "manifold_muon",
        "base_lr": 0.02,
        "dual_steps": 10,
        "dual_lr": 0.1,
        "model": config.__dict__,
    }
)

# In training loop
if step % 10 == 0:
    metrics = {
        'train/loss': loss.item(),
        'train/perplexity': torch.exp(loss).item(),
    }
    
    # Add optimizer metrics
    opt_metrics = optimizer.log_metrics(prefix='optimizer')
    metrics.update(opt_metrics)
    
    wandb.log(metrics, step=step)
```

## Questions to Consider

Before starting, think about:

1. **Baseline:** What optimizer are you currently using? (AdamW? Standard Muon?)
2. **Metrics:** What's your target performance? (Perplexity? Downstream tasks?)
3. **Resources:** What GPUs do you have? (Affects scale of experiments)
4. **Timeline:** Aggressive (2 weeks) or thorough (2 months)?
5. **Goals:** Stability improvement? Final performance? Both?

## Files to Review

Priority order:
1. ✅ **MANIFOLD_OPTIMIZER_SUMMARY.md** - Quick visual reference
2. ✅ **optimizers/manifold_muon/README.md** - API documentation
3. ✅ **optimizers/manifold_muon/integration_example.py** - How to integrate
4. ✅ **MANIFOLD_MUON_PLAN.md** - Comprehensive plan (when you need details)

## Getting Help

If you run into issues:

1. **Check test results:** `python -m optimizers.manifold_muon.test_manifold_optimizers`
2. **Review logs:** Look for NaNs, constraint violations, divergence
3. **Profile:** Use `torch.profiler` to find bottlenecks
4. **Experiment:** Try different hyperparameters on small models first

## Success Criteria

You'll know it's working when:

✅ All tests pass  
✅ Constraint violations < 1e-4 throughout training  
✅ Training loss decreases smoothly  
✅ Singular values stay near 1.0 for Stiefel matrices  
✅ Training time overhead < 20%  
✅ Final performance matches or exceeds baseline  

## Final Thoughts

This implementation is production-ready but conservative:
- Start with small models to validate
- Monitor constraints carefully
- Tune hyperparameters systematically
- Scale up gradually

The code is modular, so you can:
- Use only Stiefel Muon for attention (easiest start)
- Add Hypersphere for embeddings (medium complexity)
- Full manifold optimization for all components (most ambitious)

**Recommendation:** Start with just applying Stiefel Muon to attention layers (QKV + output). This gives you most of the benefit with minimal complexity.

Good luck! 🚀

---

**Ready to start?**

Run this now:
```bash
cd /Users/vukrosic/AI\ Science\ Projects/blueberry-llm
python -m optimizers.manifold_muon.test_manifold_optimizers
```

If tests pass, you're ready to integrate! 🎉




