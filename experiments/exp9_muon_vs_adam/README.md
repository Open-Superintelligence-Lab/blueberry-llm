# Experiment 9: Muon vs Adam Optimizer Comparison

## Overview

This experiment systematically compares the **Muon optimizer** (MomentUm Orthogonalized by Newton-schulz) against the standard **Adam/AdamW optimizer** for training Mixture-of-Experts (MoE) transformer models with full attention.

## Motivation

Different optimizers can significantly impact training dynamics, convergence speed, and final model performance. This experiment aims to:

1. **Compare convergence speed**: Does Muon converge faster than Adam?
2. **Compare final performance**: Which optimizer achieves better validation loss?
3. **Test learning rate sensitivity**: How do different learning rates affect each optimizer?
4. **Evaluate training stability**: Are there differences in training stability between optimizers?

## Experiment Design

### Model Architecture
- **Type**: MoE Transformer with full attention (classic architecture)
- **Experts**: 8 experts with top-2 routing
- **Dimensions**: d_model=384, n_heads=8, n_layers=6, d_ff=1536
- **Parameters**: ~79M total (~28.4% active per forward pass)

### Experiments

#### Core Comparison
1. **muon_baseline**: Hybrid Muon + AdamW (Muon for 2D weights, AdamW for embeddings/norms)
2. **adam_baseline**: Pure AdamW for all parameters

#### Learning Rate Ablation
3. **adam_higher_lr**: Adam with LR=0.002
4. **adam_lower_lr**: Adam with LR=0.0005

#### Additional Tests
5. **muon_only**: Pure Muon with higher LR (0.02)
6. **muon_constant_lr**: Muon without LR schedule
7. **adam_constant_lr**: Adam without LR schedule

### Training Configuration
- **Steps**: 1,000 training steps
- **Batch size**: 24
- **Gradient accumulation**: 4 steps
- **LR schedule**: Cosine decay with warmup (5% warmup, min LR ratio 0.1)
- **Load balancing weight**: 0.01
- **Dataset**: HuggingFaceTB/smollm-corpus (cosmopedia-v2)
  - Training docs: 1,800
  - Validation docs: 200
  - Sequence length: 512 tokens

### Optimizer Details

**Muon (Hybrid)**:
- Muon optimizer for 2D weight matrices (uses Newton-Schulz orthogonalization)
- AdamW for embeddings and normalization layers
- Default Muon LR: 0.01
- Default AdamW LR: 0.001

**Adam**:
- AdamW for all parameters
- Default LR: 0.001
- Weight decay: 0.1

## Directory Structure

```
exp9_muon_vs_adam/
├── exp_configs/
│   ├── __init__.py
│   └── experiment_configs.py       # Experiment definitions
├── exp_training/
│   ├── __init__.py
│   └── experiment_trainer.py       # Custom trainer with optimizer selection
├── logs/                            # Training logs
├── run_experiments.py              # Main experiment runner
├── README.md                       # This file
├── EXPERIMENT_CARD.txt            # Quick reference card
└── [experiment results]/          # Generated during training
    ├── muon_baseline/
    │   ├── metrics.json
    │   ├── metrics_plot.png
    │   └── model.pt
    ├── adam_baseline/
    │   └── ...
    ├── comparison_plot.png         # Cross-experiment comparison
    └── comparison_summary.json     # Summary statistics
```

## Usage

### List Available Experiments
```bash
cd experiments/exp9_muon_vs_adam
python run_experiments.py --list
```

### Run Quick Comparison (Recommended First)
```bash
python run_experiments.py --quick
```
This runs `muon_baseline` and `adam_baseline` for a direct comparison.

### Run Specific Experiments
```bash
python run_experiments.py -e muon_baseline adam_baseline adam_higher_lr
```

### Run All Experiments
```bash
python run_experiments.py --all
```

### Specify Output Directory
```bash
python run_experiments.py -e muon_baseline adam_baseline -o ./results
```

## Output Files

Each experiment produces:
- **metrics.json**: Complete training history, configuration, and final metrics
- **metrics_plot.png**: 4-panel visualization (loss vs time, loss vs steps, accuracy, LR schedule)
- **model.pt**: Final model checkpoint

Cross-experiment comparison produces:
- **comparison_plot.png**: Side-by-side comparison of all runs
- **comparison_summary.json**: Statistical summary and best configurations

## Metrics Tracked

For each experiment:
- Validation loss (primary metric)
- Validation accuracy
- Validation perplexity
- Learning rate schedule
- Training time (wall-clock)
- Steps to best validation loss

## Expected Results

We expect to observe:

1. **Muon advantages**:
   - Potentially faster convergence in early training
   - Better conditioning of weight matrices
   - Possible advantage for deep networks

2. **Adam advantages**:
   - More stable training dynamics
   - Better performance with standard hyperparameters
   - Easier to tune

3. **Learning rate sensitivity**:
   - Muon may require higher learning rates (0.01-0.02)
   - Adam works well with lower learning rates (0.0005-0.002)

## Analysis Guidelines

When analyzing results, consider:

1. **Final validation loss**: Which optimizer achieves the lowest loss?
2. **Convergence speed**: Which reaches good performance faster?
3. **Training stability**: Are there loss spikes or instabilities?
4. **Computational cost**: Training time comparison (wall-clock)
5. **Hyperparameter sensitivity**: How sensitive is each optimizer to LR changes?

## References

- **Muon Optimizer**: MomentUm Orthogonalized by Newton-schulz
  - Uses Newton-Schulz iteration for gradient orthogonalization
  - Designed for improved conditioning of weight updates
  
- **Adam/AdamW**: Adaptive Moment Estimation with decoupled weight decay
  - Industry standard optimizer
  - Proven track record on transformer models

## Notes

- All experiments use the same random seed (42) for reproducibility
- Data is split before tokenization to prevent leakage
- AMP (Automatic Mixed Precision) is enabled by default
- Gradient clipping is set to 1.0 for all experiments

## Next Steps

After completing this experiment:

1. Identify the best optimizer configuration
2. Consider combining with other optimizations (e.g., different LR schedules)
3. Test on longer training runs if results are promising
4. Experiment with different model architectures

## Contact

For questions or issues with this experiment, refer to the main project documentation.

