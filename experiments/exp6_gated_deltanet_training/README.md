# Experiment 6: Gated DeltaNet Training with FLA

This experiment trains a language model using pure Gated DeltaNet layers with Flash Linear Attention (FLA) optimizations.

## Overview

Gated DeltaNet is a linear attention mechanism that provides:
- **Linear complexity**: O(n) instead of O(n²) for standard attention
- **Efficient training**: Optimized with FLA kernels
- **Competitive performance**: Achieves strong results on language modeling tasks



## Installation

Install experiment-specific dependencies:

```bash
cd experiments/exp6_gated_deltanet_training
pip install -r requirements.txt
```

Verify FLA is installed:
```bash
python -c "import fla; print(fla.__version__)"
```

## Usage

### Quick Start

```bash
# Start training
python run_experiment.py --config rtx4090

# Resume from checkpoint
python run_experiment.py --resume checkpoints/best_model.pt

# Resume and train longer
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 5000
```

### Configuration Options

Available presets via `--config`:
- **small**: ~2M params, batch=2, 1000 steps (quick testing)
- **medium**: ~15M params, batch=4, 2000 steps (default)
- **large**: ~60M params, batch=8, 5000 steps (full training)
- **xlarge**: ~200M params, batch=4, 10000 steps (very large)
- **rtx4090**: ~100M params, batch=32, 2000 steps (GPU optimized)

### Resume Training

Training automatically saves:
- `checkpoints/best_model.pt` - Best model by validation loss
- `checkpoints/checkpoint_step_*.pt` - Periodic checkpoints (every 500 steps)
- `checkpoints/final_model.pt` - Final model

Resume from any checkpoint to continue training or extend to more steps.

### Inference

Generate text from a trained model:
```bash
python inference.py
```

### Custom Configuration

Modify `config.py` or create custom configs:

```python
from experiments.exp6_gated_deltanet_training.config import ExperimentConfig

config = ExperimentConfig(
    hidden_size=384,
    num_hidden_layers=10,
    max_steps=3000,
    batch_size=8,
    learning_rate=3e-4,
)
```

## Learning Rate Ablation Study (H100)

We conducted a comprehensive learning rate ablation study on NVIDIA H100 (80GB) to find the optimal learning rate for batch_size=120:

### Results Summary

Tested 7 learning rates over 1000 steps each:

| Rank | Learning Rate | Best Val Loss | Final Val Loss | Final Accuracy |
|------|---------------|---------------|----------------|----------------|
| 🥇 1 | **1.00e-03** | **6.671** | 8.924 | 14.77% |
| 🥈 2 | 5.80e-04 | 6.822 | 8.760 | 14.29% |
| 🥉 3 | 7.00e-04 | 6.835 | 8.788 | 14.52% |
| 4 | 5.00e-04 | 6.952 | 8.780 | 13.33% |
| 5 | 3.00e-04 | 7.150 | 8.811 | 10.59% |
| 6 | 2.00e-04 | 7.146 | 8.557 | 8.71% |
| 7 | 4.00e-04 | 7.166 | 8.934 | 11.60% |

### Key Findings

- ✅ **Best LR: 1.00e-03** - Achieved lowest validation loss (6.671) during training
- 📊 Higher learning rates (7e-4 to 1e-3) significantly outperformed conservative rates
- 🎯 Our sqrt-scaled estimate (5.8e-4) performed well but slightly conservative
- ⚡ Training speed: ~2.6 steps/s on H100 at 90% memory utilization (74GB/81GB)
- 🔥 **Recommendation**: Use `learning_rate=1e-3` for production training

### Configuration

The ablation used:
- **Model**: 768d, 12 layers (100M params)
- **Batch size**: 120 (optimized for H100)
- **Sequence length**: 1024 tokens
- **GPU**: NVIDIA H100 80GB HBM3
- **Steps per experiment**: 1000

See visualization: `lr_ablation/lr_ablation_comparison.png`

### Production Training (10K Steps)

Based on ablation results, we're running production training with:
```bash
python run_experiment.py --config h100
```

**Configuration:**
- Learning Rate: 1e-3 (winner from ablation)
- Batch Size: 120
- Steps: 10,000 (warmup: 1,000)
- Expected time: ~65 minutes on H100

**Monitor progress:**
```bash
# Live monitor with GPU stats (auto-refresh every 5s)
bash watch_training.sh

# Or check log manually
tail -f training_10k.log
```

**Checkpoints saved to:** `checkpoints/best_model.pt`

See `TRAINING_10K_STATUS.md` for detailed monitoring guide.

## Model Details

### Gated DeltaNet Parameters

- `linear_num_value_heads`: Number of value heads (default: 4)
- `linear_num_key_heads`: Number of key heads (default: 4)
- `linear_key_head_dim`: Dimension per key head (default: 64)
- `linear_value_head_dim`: Dimension per value head (default: 64)
- `linear_conv_kernel_dim`: 1D convolution kernel size (default: 4)
