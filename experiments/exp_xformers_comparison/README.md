# XFormers Memory-Efficient Attention Comparison

Simple experiment comparing standard PyTorch attention vs xformers memory-efficient attention.

## Quick Start

```bash
cd /Users/vukrosic/AI\ Science\ Projects/blueberry-llm
python experiments/exp_xformers_comparison/compare_attention.py
```

## What It Measures

- **Training Time**: Total time to complete training steps
- **Peak Memory Usage**: Maximum GPU memory allocated
- **Final Validation Loss**: Model performance
- **Speedup**: How much faster (or slower) xformers is
- **Memory Reduction**: How much memory xformers saves

## Results

Results are saved to `results/comparison_results.json` with:
- Individual metrics for each attention type
- Direct comparison summary
- Full training/validation loss curves

