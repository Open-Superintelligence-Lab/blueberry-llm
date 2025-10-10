# MoE Ablation - Simple Workflow ⚡

## 🎯 Quick Start (3 Steps)

### 1. Edit your config
Open `configs_ablation.py` and modify:
```python
CUSTOM = AblationConfig(
    name="custom",
    batch_size=32,      # ← Edit this
    seq_len=384,        # ← Edit this
    lr=0.015,
    grad_accum=2,
    max_steps=50
)
```

### 2. Run it
```bash
python run_ablation.py custom
```

### 3. See results
Output shows:
- ✅ Val Loss & Accuracy
- 🚀 Throughput (tokens/sec)
- 💾 **Peak Memory** (to max out GPU)

---

## 📊 Plot All Results
```bash
python plot_results.py
```
Saves to `results/ablation_batch_seqlen/ablation_results.png`

---

## 🔧 Finding Your Max Memory Config

**Goal:** Max out your GPU memory for best throughput

**Method:**
1. Start: `python run_ablation.py quick`
2. Edit `configs_ablation.py` → increase `batch_size` or `seq_len`
3. Run: `python run_ablation.py custom`
4. Check **Peak Memory** in output
5. Repeat until OOM error → back off 10-20%

**Example progression:**
```python
# Try 1: batch=32, seqlen=384 → Peak: 8GB ✅
# Try 2: batch=64, seqlen=384 → Peak: 14GB ✅
# Try 3: batch=96, seqlen=384 → Peak: 21GB ✅
# Try 4: batch=128, seqlen=384 → OOM! ❌
# Final: batch=96, seqlen=384 → Perfect! 🎯
```

---

## 📋 Pre-made Configs

```bash
python run_ablation.py quick        # Fast test (5 steps)
python run_ablation.py large_batch  # Max batch (64×256)
python run_ablation.py long_seq     # Max seqlen (8×1024)
python run_ablation.py balanced     # Balanced (24×512)
python run_ablation.py max_batch    # Aggressive batch (128×128)
python run_ablation.py max_seq      # Aggressive seqlen (4×2048)
```

---

## 🔍 Understanding Output

```
✅ RESULTS: custom
   Val Loss: 7.7533       ← Lower is better
   Val Acc: 9.23%         ← Higher is better  
   Throughput: 36,132 tok/s  ← Higher is better
   Peak Memory: 8.45 GB   ← Use this to tune batch/seqlen
```

**Memory ~= batch_size × seq_len × model_params**

If Peak Memory < GPU Memory → increase batch or seqlen!

---

## 🎯 Tips

- **More batch** = better throughput, same learning
- **More seqlen** = better long-range context
- **Grad accum** = larger effective batch without more memory
- Start with `quick`, iterate fast

---

That's it! Edit config → Run → Check memory → Repeat 🚀

