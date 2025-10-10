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

```bash
cd experiments/exp5_simple_transformer_moe
python run_experiment.py
```

## Results

Results will be saved to `results/` directory including:
- Training metrics and loss curves
- Validation performance
- Model checkpoints
- Visualization plots

## Key Differences from Other Experiments

Unlike other experiments in this repository:
- **No sparse attention** (exp2, exp3)
- **No adaptive mechanisms** (experiment_3)
- **No GDN or custom architectures** (exp1, exp3)
- **No special indexing** (exp4)

This is intentionally the simplest effective architecture possible.

