# MoE Ablation Study - Quick Guide

## 🎯 New Simple Approach (Recommended)

### 1. Edit configs in `configs_ablation.py`
```python
CUSTOM = AblationConfig(
    name="custom",
    batch_size=32,      # ← Edit to max out GPU
    seq_len=384,        # ← Adjust based on memory
    lr=0.015,
    grad_accum=2,
    max_steps=50
)
```

### 2. Run your config
```bash
python run_ablation.py custom
```

### Available configs:
```bash
python run_ablation.py              # List all configs
python run_ablation.py quick        # Quick test (5 steps)
python run_ablation.py large_batch  # Large batch strategy
python run_ablation.py long_seq     # Long sequence strategy
python run_ablation.py balanced     # Balanced approach
python run_ablation.py max_batch    # Push batch size limit
python run_ablation.py max_seq      # Push sequence length limit
```

## 📊 Plot Results
```bash
python plot_results.py
```

## 💡 Workflow for Finding Max Memory

1. **Start with quick test:**
   ```bash
   python run_ablation.py quick
   ```

2. **Edit `configs_ablation.py` and increase batch/seqlen**

3. **Test again:**
   ```bash
   python run_ablation.py custom
   ```

4. **Check memory usage in output** (shows peak memory)

5. **Repeat until OOM, then back off slightly**

## 🔧 Model Config

Edit `config.py` to change MoE settings:
- `num_experts`: Number of experts (default: 8)
- `expert_top_k`: Experts per token (default: 2)
- `load_balancing_weight`: Load balancing loss weight (default: 0.01)

---

## Old CLI Method (still works)

<details>
<summary>Click to expand</summary>

```bash
python ablation_batch_vs_seqlen.py --batch 32 --seqlen 384 --lr 0.015
```

All args: `--batch`, `--seqlen`, `--lr`, `--grad-accum`, `--steps`, `--name`

</details>
