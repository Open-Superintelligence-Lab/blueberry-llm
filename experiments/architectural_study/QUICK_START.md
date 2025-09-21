# Quick Start Guide: Architectural Experiments

## 🚀 Running the Experiments

### Prerequisites
```bash
pip install matplotlib seaborn pandas torch
```

### Run All Experiments
```bash
python architectural_experiments.py
```

### Run Individual Baseline Comparison
```bash
python experiment_490_max.py
```

## 📊 Understanding the Results

### Key Files Generated:
- `experiment_results.png` - Comprehensive visualizations
- `correlation_heatmap.png` - Parameter correlation analysis
- `experiment_results.json` - Raw numerical data
- `detailed_analysis.txt` - Text summary of findings

### Reading the Output:
1. **Training Progress**: Watch for validation loss improvements
2. **Final Rankings**: Experiments ranked by validation loss (lower = better)
3. **Performance Metrics**: Loss, accuracy, training time comparison

## 🎯 Key Findings Summary

| Rank | Configuration | Val Loss | Improvement |
|------|---------------|----------|-------------|
| 1 | SmallBatch | 6.7803 | 🥇 Winner |
| 2 | Deeper | 7.2196 | -0.0008 |
| 3 | Baseline | 7.2204 | Reference |

## 🔧 Customizing Experiments

### Adding New Configurations:
```python
# In architectural_experiments.py, add to get_experimental_configs():
new_config = self.create_config_variant(baseline,
    d_model=384,  # Your changes here
    n_layers=10,
    # ... other parameters
)
configs.append(ExperimentConfig(
    "YourName", new_config, "Description of your experiment"
))
```

### Modifying Experiment Parameters:
- **Training Steps**: Change `max_steps` in baseline config
- **Batch Size**: Modify `batch_size` and `gradient_accumulation_steps`
- **Model Size**: Adjust `d_model`, `n_layers`, `n_heads`

## 📈 Interpreting Visualizations

### experiment_results.png:
- **Top Row**: Performance metrics (loss, accuracy, time)
- **Bottom Row**: Parameter relationships and correlations
- **Gold bars**: Best performing configurations

### correlation_heatmap.png:
- **Red**: Positive correlation
- **Blue**: Negative correlation
- **White**: No correlation
- **Focus on**: Validation loss correlations

## 🎓 Best Practices

1. **Always use same random seed** (42) for reproducibility
2. **Run multiple times** to verify results
3. **Monitor GPU memory** usage for larger models
4. **Check training stability** (loss curves should be smooth)
5. **Validate on held-out test set** for final evaluation

## 🚨 Common Issues

### Out of Memory:
- Reduce batch size
- Reduce sequence length
- Reduce model dimensions

### Training Instability:
- Lower learning rate
- Increase gradient clipping
- Add more regularization

### Poor Performance:
- Check data quality
- Verify model architecture
- Ensure proper initialization

---

*For detailed analysis, see the main README.md file.*
