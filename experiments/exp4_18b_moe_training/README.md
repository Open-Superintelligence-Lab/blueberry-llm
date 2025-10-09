# Experiment 4: 18B Parameter MoE Training

## Overview

This experiment trains an 18 billion parameter Mixture of Experts (MoE) language model optimized for NVIDIA B200 GPUs with 192GB VRAM.

## Model Architecture

### Configuration
- **Total Parameters**: ~18B
- **Active Parameters per Token**: ~4.5B (25% utilization due to sparse routing)
- **Hidden Dimension**: 4096
- **Layers**: 40
- **Attention Heads**: 32
- **FFN Dimension**: 11,008 (per expert)
- **Sequence Length**: 4096 tokens

### Mixture of Experts
- **Total Experts**: 8
- **Active Experts per Token**: 2 (top-k routing)
- **Expert Utilization**: 25% (only 2 out of 8 experts active)
- **Load Balancing**: Auxiliary loss to encourage uniform expert usage

## Memory Optimizations

### Gradient Checkpointing ✅
- Enabled on all transformer blocks
- Saves ~40-50% of activation memory
- Minimal speed impact (~10-15% slower)

### Muon Optimizer
- ~30% less memory than AdamW
- Only stores momentum buffer (8 bytes/param vs 12 bytes for AdamW)
- Better suited for large-scale training

### Mixed Precision (FP16)
- Forward/backward passes in FP16
- Master weights in FP32
- ~50% memory reduction for activations

## Expected Memory Usage

| Component | Memory (GB) |
|-----------|-------------|
| Model Weights (BF16) | ~36 GB |
| Gradients (BF16) | ~36 GB |
| Muon Momentum | ~36 GB |
| Activations (w/ checkpointing) | ~40-50 GB |
| **Total** | **~150-160 GB** |

**Headroom**: ~30-40 GB for data loading and CUDA overhead

## Training Configuration

- **Batch Size**: 4
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 32 (131,072 tokens/step)
- **Learning Rate**: 0.01 (Muon) / 0.001 (AdamW for embeddings/norms)
- **LR Schedule**: Warmup (5% steps) + Cosine decay
- **Max Steps**: 50,000
- **Gradient Clipping**: 1.0

## Key Features

### 1. Sparse Computation
Only 25% of parameters are active per forward pass due to top-2/8 expert routing. This provides:
- 4x model capacity increase
- Same compute as a 4.5B dense model
- Specialization potential (different experts learn different patterns)

### 2. Gradient Checkpointing
Implemented at the transformer block level:
- Recomputes activations during backward pass
- Trades compute for memory
- Critical for fitting 18B model in 192GB

### 3. Hybrid Optimization
- **Muon**: For all 2D weight matrices (attention, FFN)
- **AdamW**: For embeddings and layer norms
- Best of both worlds: efficiency + stability

## Running the Experiment

### Prerequisites
```bash
# Install dependencies
pip install torch torchtune matplotlib tqdm

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

### Run Training
```bash
cd experiments/exp4_18b_moe_training
python run_experiment.py
```

### Monitor Training
The script will output:
- Real-time training metrics (loss, accuracy, perplexity)
- Memory usage tracking
- Validation metrics every 500 steps
- Checkpoints every 5000 steps

### Expected Runtime
- **Tokens/sec**: ~5,000-10,000 (depends on hardware)
- **Total training time**: ~10-20 hours for 50k steps
- **Tokens processed**: ~6.5 billion tokens

## Output Files

### Checkpoints
- `checkpoints/checkpoint_step_*.pt`: Periodic checkpoints
- `checkpoints/checkpoint_latest.pt`: Latest checkpoint
- Contains: model weights, optimizer states, scheduler states, metrics

### Results
- `checkpoints/training_results.json`: Complete training metrics
- `checkpoints/training_curves.png`: Loss and perplexity plots

## Model Scaling

If you need to adjust the model size for different hardware:

### For 80GB GPUs (A100/H100)
```python
d_model: int = 2048       # ~5B total params
n_layers: int = 32
n_heads: int = 16
d_ff: int = 5504
```

### For 40GB GPUs (A100)
```python
d_model: int = 1536       # ~2B total params
n_layers: int = 24
n_heads: int = 12
d_ff: int = 4096
```

### For Larger Models (H200/B200+)
```python
d_model: int = 5120       # ~25B total params
n_layers: int = 48
n_heads: int = 40
d_ff: int = 13824
batch_size: int = 2       # Reduce batch size
```

## Monitoring GPU Memory

```python
import torch

# During training, check:
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `batch_size` (e.g., 4 → 2)
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_len` (e.g., 4096 → 2048)
4. Reduce model size (fewer layers or smaller d_model)

### Slow Training
1. Check if gradient checkpointing is needed (try disabling if memory allows)
2. Increase `batch_size` if memory available
3. Reduce `eval_steps` to evaluate less frequently
4. Use `num_workers=2` in DataLoader if CPU is bottleneck

### NaN Loss
1. Reduce learning rate
2. Increase warmup steps
3. Check for extreme gradient values
4. Verify data quality

## Research Questions

This experiment enables investigation of:
1. **Expert Specialization**: Do different experts learn different patterns?
2. **Scaling Laws**: How does MoE scaling compare to dense models?
3. **Load Balancing**: What's the optimal load balancing weight?
4. **Routing Strategies**: Is top-2 optimal, or should we use top-1/top-3?
5. **Memory-Compute Tradeoffs**: Impact of gradient checkpointing on training efficiency

## References

- **MoE**: Shazeer et al. (2017) - "Outrageously Large Neural Networks"
- **Muon Optimizer**: Muon paper - Memory-efficient optimization
- **Gradient Checkpointing**: Chen et al. (2016) - "Training Deep Nets with Sublinear Memory Cost"
- **Switch Transformers**: Fedus et al. (2021) - Large-scale MoE models

