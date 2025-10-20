# Experiment 9: Pre-Norm vs Pre+Post-Norm MoE Architecture

## Overview
This experiment compares the original MoE transformer block architecture (pre-normalization only) with a new architecture that uses both pre and post normalization, based on [PR #35](https://github.com/Open-Superintelligence-Lab/blueberry-llm/pull/35/files).

## Architecture Comparison

### Pre-Norm Only (Original)
```
x -> norm1 -> attention -> dropout -> add
x -> norm2 -> moe -> dropout -> add
```

### Pre+Post-Norm (New)
```
x -> pre_norm1 -> attention -> post_norm1 -> dropout -> add
x -> pre_norm2 -> moe -> post_norm2 -> dropout -> add
```

## Hypothesis
Adding post-normalization after the attention and MoE layers may improve training stability and convergence by better controlling the magnitude of activations flowing through residual connections.

## Running the Experiment
```bash
cd experiments/exp9_pre_post_norm
python run_experiment.py
```

## Configuration
- Model: 4-layer MoE Transformer
- Dimensions: 256 d_model, 4 heads, 1024 d_ff
- MoE: 4 experts, top-2 routing
- Training: 500 steps, batch size 8, seq len 256
- Optimizer: AdamW (lr=3e-4, wd=0.01)

## Results
Results will be saved to `results/experiment_results.json` with loss curves for both models.

