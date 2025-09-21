# Blueberry LLM Experiments

This directory contains comprehensive experiments conducted on the Blueberry LLM framework to identify optimal architectural configurations and training strategies.

## 📁 Available Studies

### 🏗️ [Architectural Study](./architectural_study/)
**Comprehensive analysis of MoE model architectures**

- **9 different configurations tested**
- **SmallBatch configuration wins** with 6.2% improvement
- **Complete visualizations and analysis**
- **Production-ready recommendations**

**Key Finding**: Smaller batch sizes with gradient accumulation significantly outperform larger batches, achieving better convergence and faster training.

## 🎯 Quick Start

To run the architectural experiments:

```bash
cd experiments/architectural_study
python architectural_experiments.py
```

## 📊 Results Summary

| Study | Best Configuration | Improvement | Key Insight |
|-------|------------------|-------------|-------------|
| Architectural | SmallBatch (8 batch, 4 accum) | 6.2% better loss | Batch size optimization crucial |

## 🔬 Methodology

All experiments follow strict fair comparison criteria:
- Same random seeds for reproducibility
- Consistent evaluation metrics
- Same hardware and software environment
- Controlled variable testing

## 📈 Visualizations

Each study includes comprehensive visualizations:
- Performance comparison charts
- Correlation analysis heatmaps
- Training progress plots
- Parameter efficiency analysis

---

*For detailed analysis, see individual study README files.*
