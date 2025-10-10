# Experiment 5: Simple Transformer with MoE and Muon

## Overview

This experiment implements the simplest possible transformer architecture with:
- **Standard Multi-Head Attention** (no fancy attention mechanisms)
- **Mixture of Experts (MoE)** for efficient scaling
- **Muon Optimizer** for improved training dynamics

## Goals

1. Establish a clean baseline transformer architecture
2. Demonstrate MoE effectiveness with standard attention
3. Validate Muon optimizer training efficiency
4. Create a reference implementation for future experiments

## Architecture

### Core Components
- **Standard Multi-Head Attention**: Classic scaled dot-product attention with RoPE
- **Mixture of Experts**: 8 experts with top-2 routing
- **Feed-Forward**: Standard MLP within each expert
- **Normalization**: RMSNorm for stable training
- **Position Encoding**: Rotary Position Embeddings (RoPE)

### Model Configuration
- **Model Dimension**: 384
- **Layers**: 6
- **Attention Heads**: 8
- **FFN Dimension**: 1536
- **Experts**: 8 (top-2 routing)
- **Sequence Length**: 512
- **Dropout**: 0.1

## Training Setup

### Optimizer
- **Muon**: For 2D parameters (attention, FFN weights)
  - Learning rate: 0.01
  - Momentum: 0.95
- **AdamW**: For other parameters (embeddings, norms)
  - Learning rate: 0.001
  - Weight decay: 0.1

### Training Details
- **Steps**: 20,000
- **Batch Size**: 24
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: FP16
- **Gradient Clipping**: 1.0
- **LR Schedule**: Cosine with warmup

## Expected Results

We expect this baseline to:
- Train stably without fancy attention mechanisms
- Show efficient parameter usage through MoE (only 2/8 experts active)
- Achieve competitive performance with Muon optimizer
- Serve as a clean comparison point for advanced architectures

## Running the Experiment

**Default (50M model):**
```bash
cd experiments/exp5_simple_transformer_moe
python run_experiment.py
```

**Scale to 1.5B params ($5 budget optimal):**
```bash
# First, find optimal config for your budget
python scale_model.py --budget 5 --compare

# Then update run_experiment.py to use config_1.5B instead of config
# Or modify config.py directly with the suggested parameters
```

## Performance Tracking

Training now includes **comprehensive tokens/sec tracking**:
- ⚡ **Progress bar**: Shows real-time tokens/sec during training
- 📊 **Evaluation logs**: Includes tokens/sec every 500 steps
- 🎯 **Milestone logs**: Reports throughput at key checkpoints
- 📈 **Visualization**: Dedicated plot showing throughput over time
- 💾 **JSON metrics**: Full tokens/sec history saved to results

## Ablation Studies

### Batch Size vs Sequence Length
Research question: **What's better to fill GPU memory with?**

```bash
python ablation_batch_vs_seqlen.py
```

This runs a comprehensive ablation comparing:
- **Large Batch (64×256)**: Maximize batch size with short sequences
- **Long Sequence (8×1024)**: Maximize sequence length with small batches  
- **Balanced (24×512)**: Standard balanced approach

Each strategy is tested with 3 different learning rates (0.005, 0.01, 0.02).

**Visualize results:**
```bash
python plot_ablation_results.py
```

**Quick validation mode (20 steps):**
For rapid testing of the ablation setup without long training:
```bash
# The script is already configured for quick testing (max_steps=20)
python ablation_batch_vs_seqlen.py
```

**Compare strategies by learning rate:**
Generate plots showing all strategies side-by-side for each learning rate:
```bash
python plot_ablation_duos.py
```

This creates 3 plots (`duo_lr005.png`, `duo_lr01.png`, `duo_lr02.png`), each comparing:
- Training loss curves for all strategies
- Validation metrics (loss and accuracy)
- Training throughput (tokens/sec)

**Interpreting duo plots:**
- Each plot focuses on a single learning rate
- Colors are consistent: Large Batch (red), Long Seq (teal), Balanced (blue)
- Look for which strategy achieves lowest loss at each LR
- Compare throughput vs performance trade-offs

### Budget Optimization

**Find the best model size for your budget:**
```bash
python scale_model.py --budget 5 --compare    # Compare H100 vs B200
python scale_model.py --budget 10 --gpu h100  # H100 specific
```

## Results

Results will be saved to `results/` directory including:
- Training metrics and loss curves
- Validation performance
- **Tokens/sec throughput metrics**
- Model checkpoints
- Visualization plots (including throughput graph)

## Key Differences from Other Experiments

Unlike other experiments in this repository:
- **No sparse attention** (exp2, exp3)
- **No adaptive mechanisms** (experiment_3)
- **No GDN or custom architectures** (exp1, exp3)
- **No special indexing** (exp4)

This is intentionally the simplest effective architecture possible.

