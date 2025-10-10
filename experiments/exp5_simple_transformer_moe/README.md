# Experiment 5: Batch Size vs Sequence Length Ablation Study

This experiment investigates the trade-off between batch size and sequence length while keeping the total number of tokens per training step constant.

## Study Design

We compare three strategies with identical computational budgets (tokens per step):

1. **Large Batch (64×256)**: Large batch size with short sequences
2. **Long Sequence (8×1024)**: Small batch size with long sequences  
3. **Balanced (24×512)**: Moderate batch size with moderate sequences

All configurations process the same number of tokens per step for fair comparison.

## Files

- `ablation_batch_vs_seqlen.py`: Main ablation study script
- `plot_ablation_results.py`: Comprehensive visualization of results
- `plot_ablation_duos.py`: Per-learning-rate comparison plots
- `data_cache/`: Cached tokenized data

## Usage

```bash
# Run the ablation study
python ablation_batch_vs_seqlen.py

# Generate visualizations
python plot_ablation_results.py
python plot_ablation_duos.py
```

