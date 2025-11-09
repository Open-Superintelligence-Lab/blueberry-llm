# Manifold Muon Optimizer Summary - Quick Reference

## Architecture Overview with Manifold Constraints

```
Input Token IDs
    ↓
┌─────────────────────────────────────────────────┐
│ TOKEN EMBEDDING                                 │
│ Manifold: Hypersphere (per vector)             │
│ Norm: ℓ₂                                        │
│ Optimizer: HypersphereMuon                      │
│ LR Scale: 1.0                                   │
└─────────────────────────────────────────────────┘
    ↓
    ↓ (multiply by sqrt(d_model))
    ↓
┌─────────────────────────────────────────────────┐
│ TRANSFORMER BLOCK × N                           │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │ RMSNorm (pre-attention)                   │ │
│  │ Optimizer: AdamW (unconstrained)          │ │
│  │ LR Scale: 10.0                            │ │
│  └───────────────────────────────────────────┘ │
│    ↓                                            │
│  ┌───────────────────────────────────────────┐ │
│  │ QKV PROJECTION                            │ │
│  │ Manifold: Stiefel (W^T W = I)            │ │
│  │ Norm: Spectral                            │ │
│  │ Optimizer: StiefelMuon + Dual Ascent     │ │
│  │ LR Scale: (layer/L) × sqrt(3×d/d)        │ │
│  └───────────────────────────────────────────┘ │
│    ↓                                            │
│  ┌───────────────────────────────────────────┐ │
│  │ ROTARY POSITIONAL ENCODING                │ │
│  │ (No learnable parameters)                 │ │
│  └───────────────────────────────────────────┘ │
│    ↓                                            │
│  ┌───────────────────────────────────────────┐ │
│  │ MULTI-HEAD ATTENTION                      │ │
│  │ (Flash Attention, no parameters)          │ │
│  └───────────────────────────────────────────┘ │
│    ↓                                            │
│  ┌───────────────────────────────────────────┐ │
│  │ ATTENTION OUTPUT PROJECTION               │ │
│  │ Manifold: Stiefel (W^T W = I)            │ │
│  │ Norm: Spectral                            │ │
│  │ Optimizer: StiefelMuon + Dual Ascent     │ │
│  │ LR Scale: (layer/L) × sqrt(d/d) = layer/L│ │
│  └───────────────────────────────────────────┘ │
│    ↓                                            │
│  (residual connection)                          │
│    ↓                                            │
│  ┌───────────────────────────────────────────┐ │
│  │ RMSNorm (pre-FFN)                         │ │
│  │ Optimizer: AdamW                          │ │
│  └───────────────────────────────────────────┘ │
│    ↓                                            │
│  ┌───────────────────────────────────────────┐ │
│  │ MOE ROUTER                                │ │
│  │ Manifold: Per-row Hypersphere             │ │
│  │ Norm: Row-wise ℓ₂                         │ │
│  │ Optimizer: HypersphereMuon (per-row)      │ │
│  │ LR Scale: 1.0                             │ │
│  └───────────────────────────────────────────┘ │
│    ↓ (top-k routing)                            │
│  ┌───────────────────────────────────────────┐ │
│  │ EXPERT × K (K experts per token)         │ │
│  │                                           │ │
│  │   ┌───────────────────────────────────┐  │ │
│  │   │ UP PROJECTION (d → d_ff)          │  │ │
│  │   │ Manifold: Stiefel                 │  │ │
│  │   │ Norm: Spectral                    │  │ │
│  │   │ Optimizer: StiefelMuon            │  │ │
│  │   │ LR Scale: (layer/L) × sqrt(d_ff/d)│  │ │
│  │   └───────────────────────────────────┘  │ │
│  │     ↓                                     │ │
│  │   ┌───────────────────────────────────┐  │ │
│  │   │ SiLU ACTIVATION                   │  │ │
│  │   └───────────────────────────────────┘  │ │
│  │     ↓                                     │ │
│  │   ┌───────────────────────────────────┐  │ │
│  │   │ DOWN PROJECTION (d_ff → d)        │  │ │
│  │   │ Manifold: Stiefel                 │  │ │
│  │   │ Norm: Spectral                    │  │ │
│  │   │ Optimizer: StiefelMuon            │  │ │
│  │   │ LR Scale: (layer/L) × sqrt(d/d_ff)│  │ │
│  │   └───────────────────────────────────┘  │ │
│  │                                           │ │
│  └───────────────────────────────────────────┘ │
│    ↓ (weighted combination)                     │
│  (residual connection)                          │
│                                                 │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ FINAL RMSNorm                                   │
│ Optimizer: AdamW                                │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ LM HEAD (tied with embedding)                   │
│ Same as Token Embedding                         │
└─────────────────────────────────────────────────┘
    ↓
Output Logits
```

---

## Optimizer Distribution

| Component Type | Count | Manifold | Optimizer | Priority |
|----------------|-------|----------|-----------|----------|
| QKV Projections | N layers | Stiefel | StiefelMuon | **HIGH** |
| Attn Output Proj | N layers | Stiefel | StiefelMuon | **HIGH** |
| Expert Up Proj | N×E (E experts) | Stiefel | StiefelMuon | **HIGH** |
| Expert Down Proj | N×E | Stiefel | StiefelMuon | **HIGH** |
| Token Embeddings | 1 table | Hypersphere | HypersphereMuon | **MEDIUM** |
| Router Gates | N layers | Per-row Sphere | HypersphereMuon | **MEDIUM** |
| RMSNorm Scales | 2N+1 | None | AdamW | **LOW** |

**Total Stiefel Parameters:** 4×N + 2×N×E = N×(4 + 2E)  
For N=12, E=8: **12 × 20 = 240 parameter groups**

---

## Quick Implementation Roadmap

### Stage 1: Core Infrastructure (Days 1-3)
```python
# File: optimizers/manifold_muon/core.py
- matrix_sign_newton_schulz()  ✓ (already have!)
- nuclear_norm()
- spectral_norm()
- project_to_tangent_stiefel()

# File: optimizers/manifold_muon/utils.py  
- compute_lr_scale()
- check_constraint_violation()
- log_manifold_metrics()
```

### Stage 2: Stiefel Muon (Days 4-7)
```python
# File: optimizers/manifold_muon/stiefel_muon.py
class StiefelMuon:
    - __init__()
    - _init_dual_variable()
    - _dual_ascent_step()
    - _compute_optimal_update()
    - step()
```

### Stage 3: Hypersphere Muon (Days 8-9)
```python
# File: optimizers/manifold_muon/hypersphere_muon.py
class HypersphereMuon:
    - __init__()
    - _project_to_tangent()
    - _retract_to_manifold()
    - step()
```

### Stage 4: Composition (Days 10-12)
```python
# File: optimizers/manifold_muon/composed_optimizer.py
class ComposedOptimizer:
    - __init__()
    - step()
    - zero_grad()
    - state_dict() / load_state_dict()

# File: models/moe_llm.py
- Add _tag_parameters() method
- Add create_manifold_optimizer() function
```

### Stage 5: Testing & Validation (Days 13-14)
```bash
# Small model test (d=256, n_layers=4, n_experts=4)
python train_moe.py --config configs/test_manifold.yaml --steps 1000

# Monitor:
- Constraint violations < 1e-4
- Training loss decreases
- No NaNs or instabilities
```

---

## Key Equations Reference

### Stiefel Manifold
**Constraint:** \( W^T W = I_n \)  
**Tangent Space:** \( A^T W + W^T A = 0 \)  
**Update:** \( A_{opt} = -\eta \cdot \text{msign}(G + 2W(\Lambda + \Lambda^T)) \)  
**Retraction:** \( W \leftarrow \text{msign}(W) \)

### Hypersphere  
**Constraint:** \( \|w\|_2 = 1 \)  
**Tangent Space:** \( a^T w = 0 \)  
**Update:** \( a_{opt} = -\eta \cdot \frac{g - w(w^T g)}{\|g - w(w^T g)\|_2} \)  
**Retraction:** \( w \leftarrow \frac{w}{\sqrt{1 + \eta^2}} \)

### Learning Rate Scaling
**Formula:** \( \text{lr}_{\text{eff}} = \text{lr}_{\text{base}} \times \frac{\ell}{L} \times \sqrt{\frac{n_{out}}{n_{in}}} \)

Where:
- ℓ = layer index (0 to L-1)
- L = total layers
- n_out = fan-out dimension
- n_in = fan-in dimension

---

## Configuration Template

```yaml
# configs/manifold_muon_config.yaml
optimizer:
  type: "composed_manifold"
  
  stiefel_muon:
    base_lr: 0.02
    momentum: 0.95
    nesterov: true
    dual_steps: 10
    dual_lr: 0.1
    ns_steps: 5
    constraint_tol: 1.0e-4
    
  hypersphere_muon:
    base_lr: 0.02
    momentum: 0.95
    mode: "per_vector"  # for embeddings
    
  adamw:
    base_lr: 0.2  # 10x higher for norms
    betas: [0.9, 0.999]
    weight_decay: 0.01
    
  lr_schedule:
    warmup_steps: 1000
    max_steps: 100000
    schedule: "cosine"
    min_lr_ratio: 0.1
    
  modular_scaling:
    enable_depth_scaling: true
    enable_ratio_scaling: true
```

---

## Expected Performance Characteristics

### Training Stability
| Metric | Baseline (AdamW) | Manifold Muon | Improvement |
|--------|------------------|---------------|-------------|
| Gradient Variance | High | Low | ↓ 2-3× |
| Loss Variance | Medium | Low | ↓ 1.5× |
| Constraint Satisfaction | N/A | < 1e-4 | ✓ |
| Singular Value Spread | Wide | Narrow | ↓ 10× |
| Attention Entropy Collapse | Common | Rare | ✓ |

### Model Quality (Expected)
| Metric | Baseline | Manifold Muon | Improvement |
|--------|----------|---------------|-------------|
| Final Perplexity | X | 0.95X - 0.98X | ↓ 2-5% |
| Convergence Speed | T steps | 0.9T - 0.95T | ↑ 5-10% |
| Hyperparameter Sensitivity | High | Low | ✓ |
| Expert Load Balance | Poor | Good | ↑ 20-30% |

### Computational Overhead
| Operation | Time (baseline) | Time (manifold) | Overhead |
|-----------|-----------------|-----------------|----------|
| Forward Pass | T | T | 0% |
| Backward Pass | T | T | 0% |
| Optimizer Step | 0.1T | 0.12T | +20% |
| **Total** | 2.1T | 2.12T | **+1%** |

*(Overhead minimized by infrequent dual ascent and efficient Newton-Schulz)*

---

## Troubleshooting Guide

### Problem: Constraint Violation Growing
**Symptoms:** `||W^T W - I||_F > 0.01`  
**Diagnosis:**
```python
# Check singular values
U, S, V = torch.svd(W)
print(f"Singular values: {S}")
# Should be close to 1.0
```
**Solutions:**
1. Increase `ns_steps` from 5 to 7
2. Use higher precision for Newton-Schulz (fp32 instead of fp16)
3. Check for numerical instability in gradients

### Problem: Dual Ascent Not Converging
**Symptoms:** Dual objective oscillates or increases  
**Diagnosis:**
```python
# Monitor dual objective
dual_obj = -lr * torch.sum(torch.svd(G + 2*W@(Lambda+Lambda.T)).S)
```
**Solutions:**
1. Reduce `dual_lr` from 0.1 to 0.05
2. Increase `dual_steps` from 10 to 20
3. Add momentum to dual ascent

### Problem: Training Slower Than Expected
**Symptoms:** >20% wall-clock time increase  
**Solutions:**
1. Reduce `dual_steps` from 10 to 5
2. Run dual ascent every K steps instead of every step
3. Use `torch.compile` on manifold operations
4. Consider approximate manifold updates

### Problem: Loss Diverging
**Symptoms:** Loss increases or NaN  
**Solutions:**
1. Reduce `base_lr` by 2-5×
2. Increase warmup steps
3. Check gradient clipping (manifold-aware)
4. Verify constraint initialization (start on manifold!)

---

## Validation Experiments

### Experiment 1: Constraint Satisfaction
**Goal:** Verify manifold constraints maintained  
**Method:** 
```python
for step in range(1000):
    loss = train_step(...)
    if step % 10 == 0:
        check_all_constraints(model)
```
**Success Criteria:** All constraints < 1e-4 throughout training

### Experiment 2: Ablation Study
**Goal:** Quantify benefit of each component  
**Variants:**
1. Baseline: AdamW everywhere
2. Stiefel only on attention projections
3. Stiefel on all linear layers
4. Stiefel + Hypersphere embeddings
5. Full manifold Muon (all components)

**Metric:** Final validation perplexity after 10K steps

### Experiment 3: Scalability
**Goal:** Test across model sizes  
**Configs:**
- Small: d=256, n_layers=4, n_experts=4
- Medium: d=512, n_layers=8, n_experts=8  
- Large: d=1024, n_layers=12, n_experts=16

**Success Criteria:** Consistent relative improvement across scales

### Experiment 4: Transfer Learning
**Goal:** Test if hyperparameters transfer  
**Method:** 
1. Tune on small model
2. Apply same config to medium model
3. Compare to separately tuned baseline

**Success Criteria:** <5% performance gap vs optimal tuning

---

## Next Steps Checklist

- [ ] Review and approve this plan
- [ ] Set up experiment tracking (Weights & Biases)
- [ ] Create git branch: `feature/manifold-muon`
- [ ] Implement Phase 1: Core operations
- [ ] Write unit tests for core ops
- [ ] Implement Phase 2: Stiefel Muon
- [ ] Test on toy problem (2D spiral)
- [ ] Implement Phase 3: Hypersphere Muon
- [ ] Implement Phase 4: Composition
- [ ] Integration test on small model
- [ ] Run validation experiments
- [ ] Hyperparameter tuning
- [ ] Scale to full model
- [ ] Document results
- [ ] Write blog post / paper

---

## Questions for Discussion

1. **Prioritization:** Which components should we tackle first? (Recommendation: QKV + Attn Output)

2. **Computational Budget:** How much overhead is acceptable? (10%? 20%?)

3. **Baseline:** Compare against standard Muon or AdamW? (Recommend both)

4. **Metrics:** What downstream tasks matter most? (Perplexity? HellaSwag? ARC?)

5. **Scale:** Start with small (256d) or medium (512d) model?

6. **Timeline:** Aggressive (2 weeks) or thorough (2 months)?

---

**Ready to implement? Let's start with Phase 1!** 🚀




