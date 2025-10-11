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

## Model Details

### Gated DeltaNet Parameters

- `linear_num_value_heads`: Number of value heads (default: 4)
- `linear_num_key_heads`: Number of key heads (default: 4)
- `linear_key_head_dim`: Dimension per key head (default: 64)
- `linear_value_head_dim`: Dimension per value head (default: 64)
- `linear_conv_kernel_dim`: 1D convolution kernel size (default: 4)
