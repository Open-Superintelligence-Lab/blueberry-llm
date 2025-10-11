# Full 10K Training Status

## 🚀 Training Configuration

Based on LR ablation study results, running production training with:

- **Learning Rate**: 1e-3 (best from 7 LR ablation)
- **Model**: 768d, 12 layers (~100M params)
- **Batch Size**: 120 (H100 optimized)
- **Sequence Length**: 1024 tokens
- **Total Steps**: 10,000
- **Warmup Steps**: 1,000 (10% of training)
- **GPU**: NVIDIA H100 80GB HBM3
- **Expected Time**: ~65 minutes (2.6 steps/s)

## 📊 LR Ablation Results (Already Completed)

| Rank | Learning Rate | Best Val Loss |
|------|---------------|---------------|
| 🥇 1 | **1.00e-03** | **6.671** ← Using this! |
| 🥈 2 | 5.80e-04 | 6.822 |
| 🥉 3 | 7.00e-04 | 6.835 |

See full ablation report in README.md

## 🔍 Monitor Progress

### Quick Status Check
```bash
cd /root/blueberry-llm/experiments/exp6_gated_deltanet_training
tail -20 training_10k.log
```

### Live Training Monitor
```bash
# Watch training live with GPU stats
bash watch_training.sh
```

### Check Latest Metrics
```bash
# Last training step
grep "Step.*Loss" training_10k.log | tail -1

# Last evaluation
grep -A 3 "Evaluation at step" training_10k.log | tail -4

# Best model saved
grep "New best" training_10k.log | tail -1
```

## 📁 Output Files

### Checkpoints (saved in `checkpoints/`)
- `best_model.pt` - Best model by validation loss (auto-updated)
- `checkpoint_step_*.pt` - Periodic checkpoints every 500 steps
- `final_model.pt` - Final model after 10k steps

### Results (saved in `results/`)
- `training_results.json` - Full training metrics and config
- `training_curves.png` - Loss, accuracy, perplexity plots

### Logs
- `training_10k.log` - Full training output

## ⏱️ Estimated Timeline

- **Start**: Step 0
- **Warmup**: Steps 0-1000 (LR ramps from 0 to 1e-3)
- **Main Training**: Steps 1000-10000 (LR decays)
- **Evaluations**: Every 100 steps (20 batches per eval)
- **Total Time**: ~65 minutes

Progress tracking:
- 1000 steps = ~6.5 minutes
- 2500 steps = ~16 minutes (25%)
- 5000 steps = ~32 minutes (50%)
- 7500 steps = ~48 minutes (75%)
- 10000 steps = ~65 minutes (100%) ✓

## 🎯 Expected Results

Based on ablation (1000 steps achieved 6.671 val loss):
- Should see continued improvement beyond ablation
- Target: < 6.5 validation loss
- Perplexity: < 600
- Accuracy: > 15%

## 📈 After Training Completes

The script will automatically:
1. Save final model to `checkpoints/final_model.pt`
2. Generate training curves plot
3. Save metrics to `results/training_results.json`
4. Print summary with best validation loss

Then you can:
1. Run inference: `python inference.py`
2. Resume for more training: `python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 20000`
3. Evaluate on benchmarks: `python benchmark_hellaswag.py`

