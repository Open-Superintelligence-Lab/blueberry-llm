# Experiment 8: Reasoning Architecture

**Architecture:** MoE (Mixture of Experts) + Optional Recursive Reasoning  
**Status:** ✅ All bugs fixed, ablations complete

## Final Results (1000 steps)

**🏆 Winner: Baseline** - 15.9% better than recursive reasoning

| Metric | Best Baseline | Best Reasoning |
|--------|---------------|----------------|
| **Val Loss** | **4.89** ✅ | 5.81 |
| **Perplexity** | **133** ✅ | 335 |
| **Accuracy** | **27.2%** ✅ | 20.3% |
| **Speed** | **8.3 steps/s** ✅ | 5.3 steps/s |

### Configurations
- **Baseline:** LR=6e-4, no warmup, no recursion
- **Reasoning:** LR=3e-4, warmup=100, H=1 L=1 (minimal)

### Key Findings
- ✅ **Baseline wins decisively** for 1000-step training
- ✅ **Higher LR (6e-4)** critical for fast convergence
- ✅ **60% faster training** without recursive overhead
- ⚠️ **Recursive needs 2000+ steps** to potentially show benefits
- 📊 **All 30 ablations completed** (15 baseline + 15 reasoning)

## Ablation Studies

### Baseline Ablations (15 configs)
Tested: LR, warmup, dropout, model size, MoE config, gradient clipping  
**Range:** 6.67 (best) to 9.46 (worst) val loss

### Reasoning Ablations (15 configs)
Tested: H/L cycles, ACT settings, exploration, learning rates  
**Range:** 7.59 (best) to 8.45 (worst) val loss

## Quick Start

```bash
# Train best baseline (fast)
python run_experiment.py --model-type baseline

# Train best reasoning
python run_experiment.py --model-type recursive

# Compare both
python run_experiment.py --compare

# Run ablations
python run_ablations.py              # Baseline ablations
python run_reasoning_ablations.py    # Reasoning ablations
```

## Files
- `run_experiment.py` - Main training script
- `run_ablations.py` - Baseline ablations (15 configs)
- `run_reasoning_ablations.py` - Reasoning ablations (15 configs)
- `config.py` - Default configurations
- `config_optimal.py` - Optimal configs from ablations
- `models.py` - Model architecture
- `recursive_reasoning.py` - Recursive reasoning wrapper

## Results
- `ablation_results/` - Baseline ablation results
- `reasoning_ablation_results/` - Reasoning ablation results
- `BUG_FIXES.md` - Bugs fixed
- `ABLATION_SUMMARY.md` - Detailed analysis
