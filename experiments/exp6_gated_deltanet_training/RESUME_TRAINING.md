# Resume Training Guide

This guide explains how to continue training from a saved checkpoint.

## Features

✅ **Resume from any checkpoint** - Continue training from where you left off
✅ **Automatic state restoration** - Restores model, optimizer, scheduler, and training history
✅ **Extend training** - Add more training steps beyond the original max_steps
✅ **Periodic checkpoints** - Automatic checkpoints saved every `save_interval` steps
✅ **Best model tracking** - Automatically saves the best model based on validation loss

## Quick Start

### 1. Start Normal Training
```bash
python run_experiment.py --config rtx4090
```

This will save:
- `checkpoints/best_model.pt` - Best model by validation loss
- `checkpoints/checkpoint_step_500.pt` - Periodic checkpoint at step 500
- `checkpoints/checkpoint_step_1000.pt` - Periodic checkpoint at step 1000
- `checkpoints/checkpoint_step_1500.pt` - Periodic checkpoint at step 1500
- `checkpoints/checkpoint_step_2000.pt` - Periodic checkpoint at step 2000
- `checkpoints/final_model.pt` - Model at end of training

### 2. Resume Training
```bash
# Resume from best model
python run_experiment.py --resume checkpoints/best_model.pt

# Resume from a specific checkpoint
python run_experiment.py --resume checkpoints/checkpoint_step_1000.pt
```

### 3. Resume and Extend Training
```bash
# Continue training to 5000 steps (from 2000)
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 5000

# Continue training to 10000 steps
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 10000
```

## Configuration Options

```bash
# Available model configurations
--config small      # ~2M parameters, batch=2, for quick testing
--config medium     # ~15M parameters, batch=4, default config
--config large      # ~60M parameters, batch=8, full training
--config xlarge     # ~200M parameters, batch=4, very large model
--config rtx4090    # ~100M parameters, batch=32, optimized for RTX 4090
```

## What Gets Restored

When you resume from a checkpoint, the following are restored:
- ✅ **Model weights** - Exact model state
- ✅ **Optimizer state** - Adam momentum, variance, etc.
- ✅ **Learning rate scheduler** - Current LR and warmup state
- ✅ **Training step** - Continue from exact step number
- ✅ **Best validation loss** - Track improvements
- ✅ **Training history** - Loss curves and metrics

## Example Workflow

```bash
# 1. Train initial model
python run_experiment.py --config medium

# 2. Model trained to 2000 steps, but you want more training
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 5000

# 3. Still want more? Extend again!
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 10000
```

## Checkpoint Format

Each checkpoint contains:
```python
{
    'model_state_dict': ...,      # Model weights
    'optimizer_state_dict': ...,  # Optimizer state
    'scheduler_state_dict': ...,  # LR scheduler state
    'global_step': 2000,           # Current training step
    'epoch': 0,                    # Current epoch
    'best_val_loss': 0.3961,       # Best validation loss so far
    'config': ExperimentConfig,    # Model configuration
    'train_history': [...],        # Training loss history
    'val_history': [...],          # Validation metrics history
}
```

## Tips

1. **Always resume with the same data** - The tokenizer and data should match
2. **Use --extend-steps** - When resuming from a completed run
3. **Monitor validation loss** - The best model is automatically saved
4. **Periodic checkpoints** - Check `save_interval` in config (default: 500 steps)
5. **Recovery from crashes** - Just resume from the latest checkpoint

## Troubleshooting

### "Checkpoint not found"
Make sure the path is correct:
```bash
ls checkpoints/
python run_experiment.py --resume checkpoints/best_model.pt
```

### "Training completed immediately"
You resumed from a checkpoint that already reached max_steps. Use `--extend-steps`:
```bash
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 5000
```

### "Config mismatch"
When resuming, the config from the checkpoint is used automatically. The `--config` flag is ignored.
