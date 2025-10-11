# Experiment 6: Gated DeltaNet Training with FLA

This experiment trains a language model using pure Gated DeltaNet layers with Flash Linear Attention (FLA) optimizations.

## Overview

Gated DeltaNet is a linear attention mechanism that provides:
- **Linear complexity**: O(n) instead of O(n²) for standard attention
- **Efficient training**: Optimized with FLA kernels
- **Competitive performance**: Achieves strong results on language modeling tasks
- **Recurrent inference**: Can be used in recurrent mode during inference

## Architecture

The model uses Qwen3-Next architecture with:
- **All layers**: Gated DeltaNet (linear_attention)
- **No full attention**: Pure linear attention throughout
- **Optional MoE**: Can enable Mixture-of-Experts for FFN layers

### Key Components

1. **Gated DeltaNet Layer**:
   - Query, Key, Value projections
   - 1D convolution for local context
   - Gated mechanism with learned gates
   - RMS normalization

2. **FLA Optimizations**:
   - Chunk-based processing
   - Optimized CUDA kernels
   - Memory-efficient implementations

## Installation

FLA has been installed automatically. Verify with:

```bash
python -c "import fla; print(fla.__version__)"
```

## Usage

### Quick Start

Run the training script with default configuration:

```bash
cd /root/blueberry-llm/experiments/exp6_gated_deltanet_training
python run_experiment.py
```

### Configuration Options

The experiment provides three preset configurations:

1. **Small** (Quick Testing):
   - 128 hidden size, 4 layers
   - ~1M parameters
   - 1000 training steps

2. **Medium** (Default):
   - 256 hidden size, 8 layers
   - ~10M parameters
   - 2000 training steps

3. **Large** (Full Training):
   - 512 hidden size, 12 layers
   - ~50M parameters
   - 5000 training steps

Edit `run_experiment.py` to choose configuration:

```python
# Choose one:
config = get_small_config()   # Fast testing
config = get_medium_config()  # Default
config = get_large_config()   # Full training
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

### Training Parameters

- **Learning Rate**: 3e-4 with warmup
- **Warmup Steps**: 100
- **Weight Decay**: 0.1
- **Gradient Clipping**: 1.0
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95)

## Expected Results

Training a medium model (256d, 8 layers) for 2000 steps:

- **Training Time**: ~15-20 minutes (with GPU)
- **Final Train Loss**: ~3.5-4.0
- **Final Val Loss**: ~3.8-4.2
- **Val Accuracy**: ~25-30%
- **Val Perplexity**: ~45-65

## Outputs

The experiment generates:

1. **Training Results** (`results/training_results.json`):
   - Configuration summary
   - Model architecture info
   - Training metrics
   - Best validation loss

2. **Training Curves** (`results/training_curves.png`):
   - Training loss over time
   - Validation loss
   - Validation accuracy
   - Validation perplexity

## FLA Integration

The model automatically uses FLA optimizations if available:

- **Chunk-based gated delta rule**: Efficient parallel training
- **Recurrent gated delta rule**: Fast inference
- **Causal convolution**: Optimized 1D convolution

If FLA is not available, it falls back to PyTorch implementations.

## Comparison with Other Attention

### vs. Full Attention
- ✓ Linear complexity (O(n) vs O(n²))
- ✓ Lower memory usage
- ✓ Faster for long sequences
- ✗ Slightly lower quality on some tasks

### vs. Other Linear Attention
- ✓ Gating mechanism for better expressiveness
- ✓ Local convolution for near-token interactions
- ✓ Competitive with state-of-the-art linear attention
- ✓ Optimized FLA kernels

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `max_seq_len`
- Use gradient checkpointing (add to config)

### Slow Training
- Ensure FLA is installed and working
- Check CUDA is available
- Reduce `num_hidden_layers` for testing

### Poor Performance
- Increase `max_steps`
- Tune `learning_rate`
- Increase model size
- Check data quality

## References

1. **Gated DeltaNet**: "Gated Delta Networks" - Qwen3-Next architecture
2. **FLA**: https://github.com/fla-org/flash-linear-attention
3. **Linear Attention**: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"

## Next Steps

After training:

1. **Evaluate** on downstream tasks
2. **Compare** with full attention baseline (exp1)
3. **Experiment** with hybrid architectures
4. **Scale up** to larger models
5. **Fine-tune** on specific domains

## License

Same as parent project (see root LICENSE file).

