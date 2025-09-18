# AMP vs FP32 Experiments

This directory contains the experimental framework for testing when Automatic Mixed Precision (AMP) is beneficial on T4 GPUs.

## 🚀 Quick Start

### For Google Colab:
1. Upload the entire `blueberry-llm` repository to Colab
2. Run: `!python blueberry-llm/article/experiments/setup_colab.py`
3. Run: `!python blueberry-llm/article/experiments/quick_test.py`
4. If test passes: `!python blueberry-llm/article/experiments/amp_experiment_runner.py --full`
5. Analyze: `!python blueberry-llm/article/experiments/analyze_results.py --all`

### For Local Environment:
```bash
# From the blueberry-llm root directory
cd article/experiments

# Quick test (10 steps each)
python quick_test.py

# Full experiments (1000 steps each) 
python amp_experiment_runner.py --full

# Analyze results
python analyze_results.py --all
```

## 📁 Files

- **`setup_colab.py`** - Setup script for Colab environment
- **`quick_test.py`** - Quick 10-step test to verify everything works
- **`amp_experiment_runner.py`** - Main experiment runner
- **`analyze_results.py`** - Results analysis and visualization
- **`colab_notebook.py`** - Step-by-step Colab instructions

## 🧪 Experimental Design

**Test Matrix (18 experiments):**
- 3 Model Sizes: 128d×2L, 256d×4L, 384d×6L
- 3 Batch Sizes: 4, 8, 12
- 2 Precision Modes: FP32, AMP

**Metrics Tracked:**
- Training speed (tokens/second)
- Memory usage (peak GPU memory)
- Model quality (validation loss)
- Convergence behavior

## 📊 Expected Outputs

- **Performance charts**: Speed comparison, memory usage
- **AMP benefit heatmaps**: When AMP is faster vs slower
- **Detailed results tables**: All metrics for each configuration
- **Key findings summary**: Recommendations for T4 usage

## ⏱️ Runtime

- **Quick test**: 5-10 minutes
- **Full experiments**: 30-60 minutes
- **Analysis**: 1-2 minutes

## 🎯 Expected Results

- Small models: AMP might be slower (overhead > benefits)
- Large models: AMP should be faster (memory savings)
- Large batches: AMP advantage increases
- Memory savings: ~50% reduction with AMP
