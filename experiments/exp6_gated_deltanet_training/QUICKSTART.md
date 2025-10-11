# Exp6: Gated DeltaNet Training - Quick Start Guide

## What is this experiment?

This experiment trains a language model using **Gated DeltaNet**, a linear attention mechanism (O(n) complexity) implemented with FLA (Flash Linear Attention) for optimized performance.

## Prerequisites

✅ FLA is already installed on this branch
✅ CUDA GPU with bfloat16 support (Ampere or newer recommended)

## Quick Start

### 1. 🚀 Run with RTX 4090 optimized settings (RECOMMENDED)

```bash
cd /root/blueberry-llm/experiments/exp6_gated_deltanet_training
python run_experiment.py
```

This will:
- Train a ~100M parameter DeltaNet model
- Use 768 hidden size, 12 layers, **batch size 32**
- Train for 2000 steps (~15-20 minutes with RTX 4090)
- **70-95% GPU utilization** (vs 5-10% with default configs)
- Generate training curves and save results

**Check GPU usage**: Run `watch -n 1 nvidia-smi` in another terminal to see your GPU actually working!

### 2. Run a quick test (Small model, 1000 steps)

⚠️ **Note**: Small configs have LOW GPU utilization by design (for testing only)

Edit `run_experiment.py` line 290:

```python
config = get_small_config()  # Uncomment this line
# config = get_rtx4090_optimized_config()  # Comment this line
```

Then run:
```bash
python run_experiment.py
```

This is faster (~5 minutes) but **underutilizes your GPU** (only 5-10% usage).

### 3. Other configurations

Edit `run_experiment.py` to choose:

```python
# Choose one:
config = get_small_config()                # ~2M params, testing only
config = get_medium_config()               # ~15M params, still underutilizes GPU
config = get_large_config()                # ~60M params, better GPU usage
config = get_rtx4090_optimized_config()    # ~100M params, OPTIMAL for RTX 4090 ✅
```

## Results

After training, check:
- `results/training_results.json` - Metrics and config
- `results/training_curves.png` - Loss, accuracy, perplexity plots

## Model Configurations

| Config | Parameters | Hidden Size | Layers | Batch | GPU Util | Training Time (4090) |
|--------|-----------|-------------|--------|-------|----------|---------------------|
| Small  | ~2M       | 128         | 4      | 2     | 5-10% ⚠️  | ~5 min              |
| Medium | ~15M      | 256         | 8      | 4     | 10-20% ⚠️ | ~10 min             |
| Large  | ~60M      | 512         | 12     | 8     | 40-60%   | ~25 min             |
| **4090 Opt** | **~100M** | **768** | **12** | **32** | **70-95%** ✅ | **~15 min** |
| XLarge | ~200M     | 1024        | 16     | 4     | 50-70%   | ~45 min             |

⚠️ **Small/Medium configs severely underutilize your RTX 4090!** Use 4090 Optimized config for best performance.

## Key Features

✅ **FLA Optimized**: Uses Triton-optimized kernels for speed
✅ **Linear Complexity**: O(n) instead of O(n²) attention
✅ **bfloat16 Training**: Memory-efficient mixed precision
✅ **Automatic Evaluation**: Tracks loss, accuracy, perplexity
✅ **Visualization**: Auto-generates training curves

## Troubleshooting

### OOM (Out of Memory)
Reduce batch size in `config.py`:
```python
batch_size: int = 2  # Reduce from 4
```

### Slow training
- Ensure CUDA is available: `torch.cuda.is_available()` should be True
- Check bfloat16 support: `torch.cuda.is_bf16_supported()` should be True
- Reduce sequence length for faster iterations

### Poor results
- Increase `max_steps` for longer training
- Try different learning rates
- Increase model size

## Next Steps

After successful training:

1. **Compare with baselines**: Check exp1 for comparison with full attention
2. **Evaluate on downstream tasks**: Use the trained model for specific tasks
3. **Scale up**: Try larger configurations
4. **Experiment**: Modify hyperparameters in `config.py`

## Technical Details

**What is Gated DeltaNet?**
- Linear attention mechanism with O(n) complexity
- Uses gating for better expressiveness
- Chunk-based parallel training
- Recurrent inference capability

**Why FLA?**
- Optimized Triton kernels (2-5x faster than PyTorch)
- Fused operations (normalization, cross entropy)
- Production-ready implementations
- Well-tested and maintained

## References

- FLA GitHub: https://github.com/fla-org/flash-linear-attention
- DeltaNet Paper: Check FLA documentation
- Full README: See `README.md` in this directory

