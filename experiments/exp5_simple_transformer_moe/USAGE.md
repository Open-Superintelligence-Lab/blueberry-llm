# MoE Ablation Study - Quick Guide

## 📊 Plot Results
```bash
python plot_results.py
```
Creates `results/ablation_batch_seqlen/ablation_results.png` with 4 subplots showing all metrics.

## 🚀 Run Experiments

### Run all ablations (9 configs)
```bash
python ablation_batch_vs_seqlen.py
```

### Run custom config
```bash
python ablation_batch_vs_seqlen.py --batch 32 --seqlen 384 --lr 0.015 --steps 50
```

**Arguments:**
- `--batch`: Batch size (e.g., 32, 64)
- `--seqlen`: Sequence length (e.g., 256, 512, 1024)
- `--lr`: Learning rate (e.g., 0.01, 0.015)
- `--grad-accum`: Gradient accumulation steps (default: 1)
- `--steps`: Training steps (default: 20)
- `--name`: Config name (default: "custom")

### Using the helper script
```bash
./run.sh all                                  # Run all ablations
./run.sh plot                                 # Plot results
./run.sh custom 32 512 0.015                  # Custom: batch=32, seqlen=512, lr=0.015
./run.sh custom 16 1024 0.01 4 50             # With grad_accum=4, steps=50
```

## 💡 Quick Examples

**Test larger batches:**
```bash
python ablation_batch_vs_seqlen.py --batch 128 --seqlen 128 --lr 0.02 --steps 30
```

**Test longer sequences:**
```bash
python ablation_batch_vs_seqlen.py --batch 4 --seqlen 2048 --lr 0.005 --grad-accum 8 --steps 30
```

**Quick test:**
```bash
python ablation_batch_vs_seqlen.py --batch 16 --seqlen 256 --lr 0.01 --steps 5
```

## 📈 Model Config (in config.py)

MoE is enabled with:
- **8 experts** 
- **Top-k=2** (each token routed to best 2 experts)
- **Load balancing weight=0.01**

Modify `config.py` to change these settings.

