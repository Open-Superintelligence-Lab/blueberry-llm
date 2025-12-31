# Experiment 10: Routing Temperature and Expert Specialization Analysis

## Overview

This experiment systematically explores how **routing temperature** affects Mixture-of-Experts (MoE) model training. Temperature controls the sharpness of the routing distribution - high temperature leads to more uniform routing (exploration), while low temperature leads to sharper, more confident routing decisions (exploitation).

## Research Questions

1. **How does routing temperature affect convergence speed and final performance?**
   - Does higher temperature lead to better exploration early in training?
   - What is the optimal temperature for final performance?

2. **How does temperature affect expert utilization?**
   - Do different temperatures lead to different load balancing?
   - Can we reduce load balancing loss with better temperature tuning?

3. **Does temperature scheduling help?**
   - Can we start with high temperature (exploration) and decrease over time (exploitation)?
   - What is the optimal temperature schedule?

4. **How do experts specialize under different temperatures?**
   - Do experts develop distinct specializations?
   - Does temperature affect the clarity of specialization patterns?

## Experiment Design

### Model Architecture
- **Type**: MoE Transformer with full attention (classic architecture)
- **Experts**: 8 experts with top-2 routing
- **Dimensions**: d_model=384, n_heads=8, n_layers=6, d_ff=1536
- **Parameters**: ~79M total (~28.4% active per forward pass)

### Experiments

#### Temperature Ablation (500 steps)
1. **temp_0.5**: Very sharp routing (strong exploitation)
2. **temp_1.0**: Standard softmax (baseline)
3. **temp_2.0**: Softer routing (more exploration)
4. **temp_5.0**: Very soft routing (maximum exploration)
5. **temp_10.0**: Nearly uniform routing

#### Temperature Scheduling (500 steps)
6. **schedule_linear**: Linear decay from 5.0 → 1.0
7. **schedule_cosine**: Cosine decay from 5.0 → 1.0
8. **schedule_exp**: Exponential decay from 5.0 → 1.0
9. **schedule_step**: Step decay: 5.0 (0-100) → 2.0 (100-300) → 1.0 (300+)

#### Extended Training (1000 steps)
10. **temp_best_long**: Best temperature from ablation, trained longer

### Training Configuration
- **Steps**: 500 (1000 for extended run)
- **Batch size**: 24
- **Gradient accumulation**: 4 steps
- **Optimizer**: Muon (hybrid) with optimal settings from exp9
  - Muon LR: 0.07
  - AdamW LR: 0.007
  - Momentum: 0.9
  - Weight decay: 0.2
- **LR schedule**: Cosine decay with 5% warmup
- **Load balancing weight**: 0.01
- **Dataset**: HuggingFaceTB/smollm-corpus (cosmopedia-v2)
  - Training docs: 1,800
  - Validation docs: 200
  - Sequence length: 512 tokens

### Metrics Tracked

For each experiment:
- **Performance metrics**:
  - Validation loss
  - Validation accuracy
  - Validation perplexity
  - Training time

- **Routing metrics**:
  - Expert utilization distribution
  - Load balancing loss
  - Routing entropy (diversity measure)
  - Expert selection confidence

- **Specialization metrics**:
  - Per-expert token type distribution
  - Expert activation patterns
  - Specialization clarity score

## Directory Structure

```
exp10_routing_temperature_specialization/
├── __init__.py
├── README.md
├── config.py                    # Experiment configurations
├── temperature_router.py        # Router with temperature control
├── temperature_moe.py           # MoE with temperature support
├── tracking_trainer.py          # Trainer with routing metrics tracking
├── run_experiment.py            # Main experiment runner
├── analyze_specialization.py   # Expert specialization analysis
├── plot_results.py              # Comprehensive visualization
├── results/                     # Generated during training
│   ├── temp_0.5/
│   │   ├── metrics.json
│   │   ├── routing_history.json
│   │   ├── expert_stats.json
│   │   └── plots/
│   ├── temp_1.0/
│   │   └── ...
│   └── ...
└── analysis/                    # Generated after all experiments
    ├── temperature_comparison.png
    ├── expert_utilization.png
    ├── specialization_analysis.png
    ├── routing_entropy.png
    └── summary_report.json
```

## Usage

### Run All Temperature Ablation Experiments
```bash
cd experiments/exp10_routing_temperature_specialization
python run_experiment.py --ablation
```

### Run Specific Temperature
```bash
python run_experiment.py --temperature 2.0
```

### Run All Scheduling Experiments
```bash
python run_experiment.py --schedules
```

### Run Complete Suite
```bash
python run_experiment.py --all
```

### Analyze Results
```bash
python analyze_specialization.py --results-dir ./results
python plot_results.py --results-dir ./results --output-dir ./analysis
```

## Expected Results

### Hypotheses

1. **Temperature 1.0 (baseline)** will provide reasonable performance but may not be optimal
2. **Slightly higher temperature (2.0-3.0)** may improve early exploration and lead to better final performance
3. **Very high temperature (10.0)** will hurt performance due to insufficient specialization
4. **Very low temperature (0.5)** may lead to premature specialization and suboptimal convergence
5. **Temperature scheduling** should combine benefits of exploration (early) and exploitation (late)

### Key Metrics to Watch

- **Final validation loss**: Primary performance indicator
- **Expert utilization entropy**: How evenly experts are used
- **Routing confidence**: How sharply the router selects experts
- **Convergence speed**: Steps to reach good performance
- **Expert specialization**: Do experts develop distinct roles?

## Analysis Tools

The experiment includes comprehensive analysis and visualization:

1. **Temperature comparison plots**:
   - Loss curves for all temperatures
   - Convergence speed comparison
   - Final performance summary

2. **Expert utilization visualizations**:
   - Expert usage distribution over training
   - Load balancing effectiveness
   - Routing entropy evolution

3. **Specialization analysis**:
   - Expert activation heatmaps
   - Token type preferences per expert
   - Specialization clarity metrics

4. **Routing dynamics**:
   - Routing confidence over time
   - Expert selection patterns
   - Temperature schedule effectiveness

## Key Contributions

This experiment will provide insights into:

1. **Optimal routing temperature** for MoE training
2. **Temperature scheduling strategies** that balance exploration and exploitation
3. **Expert specialization dynamics** under different routing regimes
4. **Load balancing effectiveness** as a function of temperature

## Notes

- All experiments use the same random seed (42) for reproducibility
- Data is split before tokenization to prevent leakage
- AMP (Automatic Mixed Precision) is enabled by default
- Routing metrics are tracked at every evaluation step
- Expert statistics are saved for offline analysis

## References

- **Switch Transformers** (Fedus et al., 2021): Load balancing in MoE
- **GShard** (Lepikhin et al., 2020): Scaling MoE models
- **Expert Choice Routing** (Zhou et al., 2022): Alternative routing strategies
- **Soft MoE** (Puigcerver et al., 2023): Soft expert assignments

---

**Created**: November 11, 2025
**Branch**: exp10-routing-temperature-analysis

