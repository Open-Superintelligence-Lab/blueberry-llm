# Training Continuation Plan

## Phase 1: 1000-Step Checkpoint (Current Run)

**Goal:** Create a solid checkpoint with the fixed dataset to resume from later.

### Configuration
```bash
python run_experiment.py --config h100_1k
```

**Settings:**
- Max steps: 1,000
- Warmup: 100 steps (10%)
- Learning rate: 1e-3 (constant after warmup)
- No decay in this phase

**Timeline:** ~6.5 minutes on H100

### Why This Approach?

1. **Quick validation** that the fixed stride dataset works correctly
2. **Checkpoint ready** for continuation without LR scheduling issues
3. **No premature decay** - save that for the longer run

## Phase 2: Resume and Extend (Future)

When you're ready to continue training, you have two strategies:

### Strategy A: Continue with Cosine Decay (Recommended)

Resume the checkpoint and extend to longer training with proper LR decay:

```bash
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 10000
```

**How LR scheduling will work:**
- The code detects max_steps changed (1000 → 10000)
- Creates a **new LR schedule** from step 1000 to 10000
- Applies cosine decay: LR goes from 1e-3 (current) → 0.1e-3 (10% of peak)
- Scheduler state is recalculated and fast-forwarded to step 1000

**Benefits:**
- Smooth continuation from step 1000
- Proper learning rate decay for longer training
- Better convergence in later stages

### Strategy B: Fresh Restart at Full LR

If you want to train even longer and reset the schedule:

```bash
# First, train 1k checkpoint
python run_experiment.py --config h100_1k

# Then start fresh long training using the checkpoint as initialization
python run_experiment.py --config h100 --resume checkpoints/best_model.pt
```

This treats the 1k checkpoint as pretrained weights but restarts LR schedule.

## LR Scheduling Details

### Current Implementation (from run_experiment.py)

The LR scheduler uses cosine decay with warmup:

```python
def lr_lambda(step):
    # Warmup phase
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    
    # Cosine decay phase
    return max(0.1, (max_steps - step) / (max_steps - warmup_steps))
```

### Phase 1 (1000 steps):
- Steps 0-100: Warmup (0 → 1e-3)
- Steps 100-1000: Constant at 1e-3
- Final LR: 1e-3

### Phase 2 (Resume to 10k):
- Step 1000: Start at 1e-3
- Steps 1000-10000: Cosine decay
- Final LR: 0.1e-3 (minimum)

## Expected Results

### After 1000 Steps (Phase 1):
- Validation loss: ~6.5-7.0 (should be better than ablation due to no overfitting)
- Model has seen all data ~1 epoch (968 unique samples)
- No signs of overfitting (unlike the old stride=1 training)

### After 10000 Steps (Phase 2):
- Validation loss: <6.0 (target)
- Model converged with proper decay
- ~10 epochs through the data

## Data Efficiency Comparison

**OLD (stride=1):**
- 990,362 samples (99.9% overlap)
- Each token seen ~1023 times per epoch
- Severe overfitting by step 1000

**NEW (stride=1024):**
- 968 unique samples (0% overlap)
- Each token seen 1 time per epoch
- Proper generalization

**Impact:**
- 1000 steps = ~1 full epoch through unique data
- 10000 steps = ~10 epochs (good for convergence)

## Monitoring Commands

### Check current training:
```bash
tail -f training_1k.log
```

### After completion, verify checkpoint:
```bash
ls -lh checkpoints/best_model.pt
```

### Resume for continuation:
```bash
# When ready for full 10k training
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 10000
```

## Summary

✅ **Phase 1** (1000 steps): Quick checkpoint with fixed data, constant LR
📈 **Phase 2** (10k steps): Resume and extend with proper LR decay
🎯 **Result**: Properly trained model without overfitting

