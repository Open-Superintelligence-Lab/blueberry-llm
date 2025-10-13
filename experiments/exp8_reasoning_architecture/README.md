# Experiment 8: Reasoning Architecture

**Architecture:** MoE (Mixture of Experts) + Recursive Reasoning  
**Status:** ✅ Complete

## Final Results (1000 steps)

**🏆 Winner: Baseline MoE** - 15.9% better than recursive reasoning

| Metric | Baseline MoE | Recursive Reasoning |
|--------|--------------|---------------------|
| **Val Loss** | **4.89** ✅ | 5.81 |
| **Perplexity** | **133** ✅ | 335 |
| **Accuracy** | **27.2%** ✅ | 20.3% |
| **Speed** | **8.3 steps/s** ✅ | 5.3 steps/s |

**Configurations:**
- **Baseline:** LR=6e-4, no warmup, MoE only
- **Reasoning:** LR=3e-4, warmup=100, H=1 L=1 (minimal recursion)

**Conclusion:** For 1000-step training, baseline MoE outperforms recursive reasoning. Recursive models need 2000+ steps to potentially show benefits.

## Reasoning Ablations

**15 configurations tested:**
- H/L cycle variations (H=1-3, L=1-3)
- ACT settings (enabled/disabled, exploration rates)
- Learning rates (1e-4 to 6e-4)
- Optimized combinations

**Best config:** R02_minimal (H=1, L=1)  
**Range:** 7.59 (best) to 8.45 (worst) val loss at 100 steps

**Key finding:** Minimal recursion wins for limited training steps.

## Quick Start

```bash
# Train best baseline
python run_experiment.py --model-type baseline

# Train best reasoning
python run_experiment.py --model-type recursive

# Run reasoning ablations
python run_reasoning_ablations.py

# Train final comparison (1000 steps)
python train_best_comparison.py
```

## Files

**Scripts:**
- `run_experiment.py` - Main training
- `run_reasoning_ablations.py` - 15 reasoning ablations
- `train_best_comparison.py` - Final comparison script

**Results:**
- `reasoning_ablation_results/` - 15 ablation configs
- `comparison_results/` - Final comparison (1000 steps)
- `FINAL_RESULTS.md` - Detailed analysis

**Models:**
- `checkpoints_best_baseline/` - Best baseline (val loss 4.89)
- `checkpoints_best_reasoning/` - Best reasoning (val loss 5.81)

**Code:**
- `config.py` - Configurations
- `models.py` - Model architecture
- `recursive_reasoning.py` - Recursive reasoning wrapper
