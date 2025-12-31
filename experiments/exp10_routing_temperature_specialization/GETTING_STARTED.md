# Getting Started with Experiment 10

## 🚀 Quick Start (3 minutes)

The fastest way to see the experiment in action:

```bash
cd /root/blueberry-llm/experiments/exp10_routing_temperature_specialization

# Run quick demo: 3 temperatures, 500 steps each (~6-9 minutes total)
bash quick_demo.sh
```

This will:
1. Run 3 experiments: temp_0.7, temp_1.0, temp_2.0
2. Generate all visualizations
3. Create analysis reports

Results will be in:
- `./results/` - Individual experiment results
- `./analysis/` - Comparative plots and analysis

## 📋 What Has Been Created

### Core Components (1,200+ lines of code)

1. **`temperature_router.py`** (250 lines)
   - Temperature-controlled routing with softmax scaling
   - Comprehensive statistics tracking
   - Routing entropy and confidence metrics

2. **`temperature_moe.py`** (200 lines)
   - MoE layer with temperature support
   - Expert activation tracking
   - Load balancing loss computation

3. **`temperature_model.py`** (180 lines)
   - Complete model with temperature-aware MoE
   - Model creation utilities
   - Parameter counting

4. **`tracking_trainer.py`** (300 lines)
   - Custom trainer with routing statistics
   - Temperature scheduling support
   - Comprehensive history tracking

5. **`run_experiment.py`** (350 lines)
   - Main experiment runner
   - Multiple experiment support
   - Data loading and preparation

6. **`plot_results.py`** (450 lines)
   - 6 comprehensive visualizations
   - Temperature comparison plots
   - Routing dynamics analysis
   - Expert utilization patterns
   - Schedule comparison
   - Summary report generation

7. **`analyze_specialization.py`** (350 lines)
   - Expert specialization analysis
   - Gini coefficient computation
   - Utilization variance analysis
   - Entropy trend analysis

8. **`config.py`** (150 lines)
   - 13 experiment configurations
   - Temperature scheduling functions
   - Configuration management

### Documentation (3,000+ words)

1. **`README.md`** - Complete experiment documentation
2. **`EXPERIMENT_SUMMARY.md`** - Comprehensive technical summary
3. **`EXPERIMENT_CARD.txt`** - Quick reference card
4. **`GETTING_STARTED.md`** - This file!

### Utilities

1. **`quick_demo.sh`** - Fast demo script
2. **`quick_test.py`** - Configuration verification

## 🎯 Experiment Overview

### What We're Studying

**Routing Temperature** controls how sharply the MoE router selects experts:

```
router_probs = softmax(logits / temperature)
```

- **Low temp (0.5)**: Sharp, confident routing → strong specialization
- **Medium temp (1.0)**: Balanced routing → baseline
- **High temp (5.0)**: Soft, exploratory routing → better load balance

### The 13 Experiments

#### Temperature Ablation (8 experiments)
- `temp_0.5` - Very sharp (exploitation)
- `temp_0.7` - Sharp
- `temp_1.0` - **Baseline**
- `temp_1.5` - Slightly soft
- `temp_2.0` - Soft (exploration)
- `temp_3.0` - Very soft
- `temp_5.0` - Nearly uniform
- `temp_10.0` - Uniform (extreme exploration)

#### Temperature Schedules (4 experiments)
- `schedule_linear` - Linear decay 5.0→1.0
- `schedule_cosine` - Cosine decay 5.0→1.0
- `schedule_exp` - Exponential decay 5.0→1.0
- `schedule_step` - Step decay 5.0→2.0→1.0

#### Extended Training (1 experiment)
- `temp_best_long` - Best temperature, 1000 steps

## 📊 What You'll Get

### Per-Experiment Outputs

Each experiment produces:
```
results/temp_1.0/
├── metrics.json          # Complete training history
│   ├── val_losses: [...]
│   ├── val_accuracies: [...]
│   ├── routing_entropies: [...]
│   ├── selection_confidences: [...]
│   ├── expert_utilizations: [...]
│   └── ... (20+ metrics)
│
├── model.pt              # Model checkpoint
└── logs/                 # Training logs
```

### Analysis Outputs

The analysis scripts generate:
```
analysis/
├── temperature_ablation_comprehensive.png
│   ├── Loss vs steps (all temps)
│   ├── Loss vs time
│   ├── Performance vs temperature
│   ├── Accuracy vs temperature
│   ├── Routing entropy vs temperature
│   ├── Accuracy evolution
│   └── Summary statistics table
│
├── routing_dynamics.png
│   ├── Routing entropy evolution
│   ├── Selection confidence evolution
│   ├── Load balancing loss trends
│   └── Temperature vs final metrics
│
├── expert_utilization.png
│   └── Per-expert utilization bars (all temps)
│
├── expert_utilization_analysis.png
│   ├── Gini coefficient vs temperature
│   ├── Utilization variance vs temperature
│   ├── Expert utilization heatmap
│   └── Statistics summary table
│
├── entropy_analysis.png
│   ├── Entropy evolution over training
│   └── Entropy change rate
│
├── schedule_comparison.png
│   ├── Loss comparison (all schedules)
│   ├── Temperature evolution
│   ├── Accuracy comparison
│   └── Final performance bars
│
├── summary_report.json
│   ├── Best results
│   ├── Temperature analysis
│   └── Schedule analysis
│
└── specialization_report.json
    ├── Per-experiment analysis
    ├── Utilization metrics
    ├── Routing entropy metrics
    └── Key insights
```

## 🎨 Sample Commands

### Run Single Experiment
```bash
python run_experiment.py --experiment temp_2.0
```

### Run Temperature Ablation (all 8 temps)
```bash
python run_experiment.py --ablation
```

### Run Temperature Schedules (all 4 schedules)
```bash
python run_experiment.py --schedules
```

### Run Everything (13 experiments)
```bash
python run_experiment.py --all
```

### Custom Temperature
```bash
python run_experiment.py --temperature 1.5
```

### List All Available Experiments
```bash
python run_experiment.py --list
```

### Generate Visualizations
```bash
# After running experiments
python plot_results.py --results-dir ./results --output-dir ./analysis
python analyze_specialization.py --results-dir ./results --output-dir ./analysis
```

## 🔬 Expected Insights

After running the experiments, you'll discover:

### 1. Optimal Temperature
- What temperature gives best validation loss?
- How sensitive is performance to temperature?
- Is temp=1.0 (default) actually optimal?

### 2. Exploration-Exploitation Trade-off
- How does temperature affect convergence speed?
- When is high temperature (exploration) beneficial?
- When is low temperature (exploitation) better?

### 3. Expert Specialization
- How does temperature affect expert utilization?
- Do different temperatures lead to different specialization patterns?
- What's the relationship between specialization and performance?

### 4. Load Balancing
- Can temperature tuning reduce load balancing loss?
- Is there a temperature that balances performance and load distribution?
- How does temperature affect the Gini coefficient?

### 5. Scheduling Strategies
- Does temperature scheduling help?
- What schedule shape is optimal?
- When should we transition from exploration to exploitation?

## 📈 Interpreting Results

### Key Metrics to Watch

**Validation Loss** (lower is better)
- Primary performance metric
- Look for U-shaped curve: optimal temperature in middle

**Routing Entropy** (context-dependent)
- High entropy = uniform routing (more exploration)
- Low entropy = sharp routing (strong specialization)
- Optimal depends on training phase

**Expert Utilization** (balanced is good)
- Ideal: all experts used equally (~12.5% each for 8 experts)
- Gini coefficient: 0 = perfect balance, higher = more inequality
- Low temperature tends to increase inequality

**Selection Confidence** (higher = sharper)
- How strongly is the top expert preferred?
- Should decrease with temperature
- Trade-off: too high = imbalance, too low = weak specialization

### Visualization Interpretation

#### Temperature Ablation Plot
Look for:
- **Optimal temperature**: Minimum of the loss curve
- **Sensitivity**: How steep is the curve? Flat = robust, steep = sensitive
- **Sweet spot**: Usually between 1.0 and 3.0

#### Routing Dynamics
Look for:
- **Entropy trends**: Does entropy decrease over training?
- **Confidence trends**: Does confidence increase over training?
- **Temperature effects**: How do different temps affect trends?

#### Expert Utilization
Look for:
- **Balance**: Are all experts used roughly equally?
- **Temperature effect**: Higher temp → better balance?
- **Specialization**: Low utilization variance = good balance

#### Schedule Comparison
Look for:
- **Best schedule**: Which achieves lowest loss?
- **Early vs late**: Does high early temp help?
- **Smooth vs discrete**: Cosine vs step schedule?

## 💡 Tips for Running Experiments

### For Quick Iteration
```bash
# Run just 3 representative temps (~10 min total)
python run_experiment.py --experiments temp_0.7 temp_1.0 temp_2.0
```

### For Comprehensive Analysis
```bash
# Run full ablation (~20-25 min)
python run_experiment.py --ablation

# Then run schedules (~10-12 min)
python run_experiment.py --schedules

# Finally, extended run with best temp (~4-5 min)
# Update temp_best_long in config.py with best temp first
python run_experiment.py --experiment temp_best_long
```

### For Custom Exploration
```bash
# Test specific temperature
python run_experiment.py --temperature 1.8

# Multiple custom temps
python run_experiment.py --experiments temp_1.5 temp_2.5 temp_3.5
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size in `MoEModelConfig`
- Reduce number of documents: `num_documents=1000` instead of 2000
- Disable AMP: `use_amp=False`

### Slow Training
- Reduce eval frequency: `eval_every=20` instead of 10
- Reduce eval steps: `eval_steps=50` instead of 100
- Use fewer workers: `num_workers=1` in data loader

### Import Errors
Make sure you're running from the experiment directory:
```bash
cd /root/blueberry-llm/experiments/exp10_routing_temperature_specialization
python run_experiment.py --list
```

### Plotting Errors
Install required packages:
```bash
pip install matplotlib seaborn numpy
```

## 📚 Next Steps

After running the experiments:

1. **Analyze Results**
   ```bash
   python plot_results.py
   python analyze_specialization.py
   ```

2. **Review Visualizations**
   - Open `analysis/*.png` files
   - Check `analysis/summary_report.json`
   - Review `analysis/specialization_report.json`

3. **Document Findings**
   - What was the best temperature?
   - How much improvement over baseline?
   - What scheduling strategy worked best?
   - Any surprising results?

4. **Extend Experiments**
   - Test longer training (5k-10k steps)
   - Try different model sizes
   - Test on different datasets
   - Combine with other techniques (from exp9, etc.)

5. **Share Results**
   - Add findings to README
   - Create summary plots
   - Document insights for future experiments

## 🎓 Learning Resources

To understand the experiment better, read:

1. **`EXPERIMENT_SUMMARY.md`** - Technical deep dive
2. **`README.md`** - Complete documentation
3. **`EXPERIMENT_CARD.txt`** - Quick reference

Key papers:
- Switch Transformers (Fedus et al., 2021) - Load balancing
- GShard (Lepikhin et al., 2020) - Scaling MoE
- Expert Choice Routing (Zhou et al., 2022) - Alternative routing

## ✅ Checklist

Before running experiments:
- [ ] Verify GPU available: `nvidia-smi`
- [ ] Check disk space: `df -h`
- [ ] Test configuration: `python quick_test.py`
- [ ] Review experiment list: `python run_experiment.py --list`

After running experiments:
- [ ] Check all experiments completed successfully
- [ ] Generate visualizations
- [ ] Review summary reports
- [ ] Document key findings
- [ ] Save important plots

## 🎉 You're Ready!

Everything is set up and ready to run. Start with:

```bash
bash quick_demo.sh
```

Then explore the results in `./analysis/`!

---

**Questions?** Check the documentation:
- `README.md` - Complete experiment documentation
- `EXPERIMENT_SUMMARY.md` - Technical details
- `EXPERIMENT_CARD.txt` - Quick reference

**Happy experimenting! 🚀**

