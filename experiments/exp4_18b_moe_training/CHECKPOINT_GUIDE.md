# Checkpoint Loading and Resume Training Guide

This guide explains how to load checkpoints and resume training for the MoE model.

## Quick Start

### Resume Training from a Checkpoint

```bash
# Resume from the latest checkpoint
python experiments/exp4_18b_moe_training/run_experiment.py \
    --gpu 4090 \
    --resume experiments/exp4_18b_moe_training/checkpoints/checkpoint_latest.pt

# Resume from a specific step checkpoint
python experiments/exp4_18b_moe_training/run_experiment.py \
    --gpu 4090 \
    --resume experiments/exp4_18b_moe_training/checkpoints/checkpoint_step_5000.pt
```

### Load Model for Inference Only

```python
from experiments.exp4_18b_moe_training.trainer_18b import load_checkpoint
from experiments.exp4_18b_moe_training.models_18b import MoE18BLLM
import torch

# Load checkpoint to get config
checkpoint = torch.load('checkpoints/checkpoint_latest.pt')
config = checkpoint['config']

# Create model
model = MoE18BLLM(config)

# Load weights (no optimizers needed for inference)
step, config, metrics = load_checkpoint(
    'checkpoints/checkpoint_latest.pt',
    model,
    device='cuda'
)

model.eval()  # Set to evaluation mode
print(f"Loaded model from step {step}")
```

## Checkpoint Structure

Each checkpoint contains:
- `step`: Training step number
- `model_state_dict`: Model weights
- `optimizer_states`: Optimizer states (for resuming)
- `scheduler_states`: Learning rate scheduler states (for resuming)
- `config`: Model configuration
- `metrics`: Training metrics at that step

## Checkpoint Locations

Checkpoints are saved in: `experiments/exp4_18b_moe_training/checkpoints/`

Two types of checkpoints are saved:
1. **Periodic checkpoints**: `checkpoint_step_<STEP>.pt` (saved every 5000 steps)
2. **Latest checkpoint**: `checkpoint_latest.pt` (always updated with the most recent state)

## Resume Training Behavior

When you resume training:
1. ✅ Model weights are restored
2. ✅ Optimizer states are restored (momentum, adaptive learning rates, etc.)
3. ✅ Learning rate scheduler is restored
4. ✅ Training resumes from the saved step number
5. ✅ Token count continues from where it left off

This ensures **seamless continuation** of training with no degradation in optimization.

## Example Use Cases

### 1. Training Interrupted? Resume It!

```bash
# Your training crashed at step 8,234?
python experiments/exp4_18b_moe_training/run_experiment.py \
    --gpu 4090 \
    --resume checkpoints/checkpoint_latest.pt
```

### 2. Load Model to Evaluate on Custom Data

```python
from experiments.exp4_18b_moe_training.load_and_use_checkpoint import load_model_for_inference

model, config, step = load_model_for_inference('checkpoints/checkpoint_step_10000.pt')

# Now use model for evaluation or inference
with torch.no_grad():
    logits, _ = model(input_tokens, return_aux_loss=False)
```

### 3. Continue Training with Different Settings

```python
# Load checkpoint, modify config, and continue training
from experiments.exp4_18b_moe_training.trainer_18b import train_18b_model, load_checkpoint
from experiments.exp4_18b_moe_training.config_4090 import MoE4090Config

# Load and modify config
config = MoE4090Config(vocab_size=49152)
config.muon_lr = 0.005  # Lower learning rate for fine-tuning

# Resume training with new config
model, metrics = train_18b_model(
    config,
    train_loader,
    val_loader,
    checkpoint_path='checkpoints/checkpoint_step_25000.pt'
)
```

## Helper Script

A helper script is provided: `load_and_use_checkpoint.py`

```bash
# Load model for inference
python experiments/exp4_18b_moe_training/load_and_use_checkpoint.py --mode inference

# Example of resuming training
python experiments/exp4_18b_moe_training/load_and_use_checkpoint.py --mode resume
```

## GPU Compatibility

Checkpoints saved with one GPU config can be loaded with another:
- A model trained on 4090 can be loaded on B200 (and vice versa)
- Just make sure the **model architecture** matches (same config file)
- The `--gpu` flag only affects NEW training runs, not loading

## Troubleshooting

### "RuntimeError: Error(s) in loading state_dict"
- Make sure the config matches the checkpoint
- Check that you're using the same model architecture

### "CUDA out of memory" when loading
- The checkpoint was trained on a larger GPU
- Use a smaller config or reduce batch size

### Training starts from step 0 even with --resume
- Check that the checkpoint path is correct
- Verify the checkpoint file exists

## Best Practices

1. **Always save `checkpoint_latest.pt`**: It's automatically saved and easy to resume from
2. **Keep periodic checkpoints**: They allow you to go back to earlier states if needed
3. **Test checkpoint loading**: After saving, verify you can load it before deleting old checkpoints
4. **Document your training runs**: Note which checkpoint corresponds to which experiment

## Advanced: Manual Checkpoint Loading

```python
import torch

# Load checkpoint manually
checkpoint = torch.load('checkpoints/checkpoint_step_5000.pt')

print(f"Checkpoint from step: {checkpoint['step']}")
print(f"Training metrics: {checkpoint['metrics']}")
print(f"Model config: {checkpoint['config']}")

# Access model weights
model_weights = checkpoint['model_state_dict']

# Access optimizer state
optimizer_state = checkpoint['optimizer_states'][0]  # Muon optimizer
```

## Summary

✅ **Resume training**: Use `--resume <checkpoint_path>`
✅ **Load for inference**: Use `load_checkpoint()` with no optimizer/scheduler
✅ **Checkpoints are saved automatically** every 5000 steps
✅ **Full training state is preserved**: weights, optimizer, scheduler, step number

For more examples, see `load_and_use_checkpoint.py`!

