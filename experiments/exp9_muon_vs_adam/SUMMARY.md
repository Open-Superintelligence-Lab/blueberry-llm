# Experiment 9 Setup Complete ✅

## Overview

Successfully created **Experiment 9: Muon vs Adam Optimizer Comparison** for training MoE models with full attention.

## What Was Created

### Directory Structure
```
experiments/exp9_muon_vs_adam/
├── exp_configs/
│   ├── __init__.py
│   └── experiment_configs.py       # 7 experiment configurations
├── exp_training/
│   ├── __init__.py
│   └── experiment_trainer.py       # Custom trainer with optimizer switching
├── logs/                            # For training logs
├── run_experiments.py              # Main experiment runner
├── view_experiments.py             # Results viewer and analyzer
├── quick_start.sh                  # Interactive menu script
├── README.md                       # Full documentation
├── EXPERIMENT_CARD.txt            # Quick reference guide
└── SUMMARY.md                      # This file
```

### Experiments Configured

1. **muon_baseline** - Hybrid Muon (2D weights) + AdamW (embeddings/norms)
   - Muon LR: 0.01, AdamW LR: 0.001
   - Cosine schedule

2. **adam_baseline** - Pure AdamW for all parameters
   - Adam LR: 0.001
   - Cosine schedule

3. **adam_higher_lr** - Adam with higher learning rate
   - Adam LR: 0.002
   - Tests sensitivity to higher LR

4. **adam_lower_lr** - Adam with lower learning rate
   - Adam LR: 0.0005
   - Tests sensitivity to lower LR

5. **muon_only** - Pure Muon with higher LR
   - Muon LR: 0.02, AdamW LR: 0.001
   - More aggressive Muon

6. **muon_constant_lr** - Muon without schedule
   - Tests Muon with constant LR

7. **adam_constant_lr** - Adam without schedule
   - Tests Adam with constant LR

## How to Use

### Quick Start (Recommended)

```bash
cd /root/blueberry-llm/experiments/exp9_muon_vs_adam
./quick_start.sh
```

This launches an interactive menu with options to:
- Run quick comparison
- Run all experiments
- List experiments
- View results

### Command Line Usage

**List all experiments:**
```bash
python run_experiments.py --list
```

**Quick comparison (2 experiments, ~20 min):**
```bash
python run_experiments.py --quick
```

**Run specific experiments:**
```bash
python run_experiments.py -e muon_baseline adam_baseline
```

**Run all experiments (~70 min):**
```bash
python run_experiments.py --all
```

**View results:**
```bash
python view_experiments.py
```

**View specific experiment:**
```bash
python view_experiments.py -e muon_baseline
```

## Expected Output

### Per Experiment
- `metrics.json` - Complete training history and configuration
- `metrics_plot.png` - 4-panel visualization
- `model.pt` - Final model checkpoint

### Comparison (when running multiple experiments)
- `comparison_plot.png` - Side-by-side comparison
- `comparison_summary.json` - Statistical summary

## Key Features

### Optimizer Implementations

**Muon (Hybrid)**:
- Uses Newton-Schulz orthogonalization for 2D weight matrices
- Falls back to AdamW for embeddings and normalization layers
- Better gradient conditioning
- May converge faster on deep networks

**Adam**:
- Standard AdamW for all parameters
- Industry standard, proven performance
- More predictable behavior

### Training Configuration
- **Model**: 79M parameter MoE with 8 experts
- **Steps**: 1,000 training steps
- **Batch size**: 24 (effective: 96 with grad accumulation)
- **Dataset**: HuggingFaceTB/smollm-corpus (cosmopedia-v2)
- **Evaluation**: Every 10 steps

### Metrics Tracked
- Validation loss (primary metric)
- Validation accuracy
- Validation perplexity
- Learning rate schedule
- Training time (wall-clock)

## Research Questions

1. ✓ Which optimizer achieves better final validation loss?
2. ✓ Which optimizer converges faster?
3. ✓ How sensitive is each optimizer to learning rate?
4. ✓ Are there stability differences during training?
5. ✓ Does hybrid Muon outperform pure Adam?

## Next Steps

After running experiments:

1. **View results**: `python view_experiments.py`
2. **Analyze comparison plot**: Check `comparison_plot.png`
3. **Review best configuration**: Check winner in console output
4. **Make decision**: Choose optimizer for future work based on results

## Documentation

- **README.md**: Full documentation with detailed explanations
- **EXPERIMENT_CARD.txt**: Quick reference card with all details
- **This file (SUMMARY.md)**: High-level overview

## Technical Details

### Path Setup
The scripts automatically handle Python path setup to import from the main project:
- Project root: `/root/blueberry-llm`
- Imports work for configs, models, optimizers, etc.

### Reproducibility
- Fixed random seed: 42
- Data split before tokenization (no leakage)
- AMP enabled for all runs
- Gradient clipping: 1.0

### Error Handling
- Experiments continue even if one fails
- Full traceback logged for debugging
- Results saved incrementally

## Testing Status

✅ All files created successfully  
✅ Directory structure verified  
✅ Import paths configured correctly  
✅ Experiment list displays properly  
✅ Scripts are executable  
✅ No linting errors  

## Support

If you encounter issues:

1. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
2. Review logs in `./logs/` directory
3. Check disk space for checkpoints and cache
4. Verify all dependencies are installed

## Ready to Run! 🚀

The experiment is fully configured and ready to run. Start with:

```bash
cd /root/blueberry-llm/experiments/exp9_muon_vs_adam
python run_experiments.py --quick
```

This will run a quick comparison between Muon and Adam baselines (~20 minutes).

---

**Created**: 2025-11-09  
**Framework**: blueberry-llm experiments v1.0  
**Experiment Type**: Optimizer comparison (Muon vs Adam)

