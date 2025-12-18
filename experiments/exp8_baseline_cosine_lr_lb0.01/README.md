# exp8_baseline_cosine_lr_lb0.01

**Status**: ✅ Completed | **Date**: 2025-11-09 | **Type**: Baseline Reference

## Description

Baseline 1000-step MoE training run that identified validation loss inflection issue at steps 850-1000.

## Configuration

- **Model**: 8 experts, top-2 routing, 79M params (28.4% active)
- **Steps**: 1000, batch size 24, grad accumulation 4
- **LR**: Cosine schedule with warmup
- **Load balancing**: 0.01
- **Dropout**: 0.1

## Results

| Metric | Best (step 950) | Final (step 1000) |
|--------|-----------------|-------------------|
| Val Loss | **5.1357** | 5.1564 |
| Val Acc | 25.20% | 25.19% |
| Val PPL | 169.98 | 173.53 |

**Time**: 3.83 minutes

## Issue Identified

⚠️ **Validation loss inflection at steps 850-1000**
- Loss stopped improving around step 850
- Degraded by 0.4% from best to final
- Suggests: overfitting, LR schedule issues, or load balancing conflicts

## Files

**Results**: `metrics.json`, `metrics_plot.png`, `val_loss_vs_time.png`  
**Code**: `run_experiments.py`, `view_experiments.py`, `exp_configs/`, `exp_training/`  
**Experiments**: Subdirectories for each experiment run (baseline/, early_stopping/, etc.)

## Usage

```bash
cd experiments/exp8_baseline_cosine_lr_lb0.01

# List experiments
python run_experiments.py --list

# Run all experiments
python run_experiments.py --all

# Run quick suite (4 key experiments)
python run_experiments.py --quick

# View results
python view_experiments.py
```

## Available Experiments

9 experiments to diagnose the inflection issue:

1. **baseline** - Reproduce current setup
2. **early_stopping** - Stop at best val loss (tests overfitting)
3. **constant_lr** - No LR schedule
4. **lower_lb_weight** - LB weight = 0.001
5. **no_lb** - No load balancing
6. **higher_dropout** - Dropout = 0.2
7. **linear_decay** - Linear LR decay
8. **slower_min_lr** - Higher min LR (0.3)
9. **short_run** - 600 steps only

## Next Steps

Run comparison experiments to diagnose:
- Overfitting: `early_stopping`, `higher_dropout`
- LR issues: `constant_lr`, `linear_decay`, `slower_min_lr`
- Load balancing: `lower_lb_weight`, `no_lb`
- Timing: `short_run`
