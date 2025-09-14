# 🫐 Getting Started with Blueberry LLM

Welcome to Blueberry LLM! This guide will get you up and running quickly.

## 🚀 Quick Start (Recommended)

### 1. Clone and Setup
```bash
git clone https://github.com/Open-Superintelligence-Lab/blueberry-llm
cd blueberry-llm
chmod +x setup.sh
./setup.sh
```

### 2. Train Your First Model
```bash
python train_auto.py
```

That's it! The auto-configuration will detect your hardware and train an optimized MoE model.

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Check Your System
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Train with Auto-Configuration
```bash
python train_auto.py
```

## 🎯 What Each Script Does

### `train_auto.py` - **START HERE**
- **Auto-detects your hardware** (CPU, GPU, memory)
- **Automatically configures** optimal model size and training parameters
- **Handles multi-GPU setup** automatically
- **Trains a MoE model** with optimal settings for your hardware

### `llm.py` - Direct Training
- Train with manual configuration
- Good for experimentation and research
- Requires manual parameter tuning

### `inference.py` - Generate Text
- Use your trained model to generate text
- Interactive chat mode available
- Requires a trained model first

## 🔍 Advanced GPU Profiler (Optional)

If you want detailed performance analysis:

### 1. Install Profiler Dependencies
```bash
pip install psutil matplotlib pandas nvidia-ml-py3
```

### 2. Train with Profiling
```bash
python train_with_profiler.py
```

### 3. Test Profiler
```bash
python test_profiler.py
```

## 📊 Expected Results

### Training Output
```
🫐 Blueberry LLM Auto-Training
==================================================
🚀 Mode: GPU Training (1 GPUs)
   Memory: 24.0 GB per GPU
📏 Model: 384d × 6L × 8H
🧠 Experts: 8
📊 Batch: 16 (accum: 2)
📝 Sequence: 1024
⚡ Mixed Precision: Yes
==================================================

🚀 Training MoE model with 8 experts (top-2)
📊 Total parameters: 1,234,567
📊 Active parameters: 456,789
📊 Parameter efficiency: 37.0% active per forward pass
```

### Inference Output
```bash
python inference.py --prompt "The future of AI is"
# Output: The future of AI is bright and full of possibilities...
```

## 🛠️ Hardware Requirements

### Minimum
- **CPU**: Any modern processor
- **RAM**: 8GB
- **Storage**: 5GB free space

### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+
- **Storage**: 20GB+ free space

### Optimal
- **GPU**: Multiple NVIDIA GPUs
- **RAM**: 32GB+
- **Storage**: 50GB+ free space

## 🚨 Troubleshooting

### Common Issues

1. **CUDA not available**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   # If False, install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Out of memory**
   ```bash
   # The auto-configuration will automatically reduce model size
   # Or manually reduce batch size in train_auto.py
   ```

3. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Permission denied on setup.sh**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## 📚 Next Steps

### For Beginners
1. Run `python train_auto.py` and let it train
2. Try `python inference.py` to generate text
3. Experiment with different prompts

### For Researchers
1. Modify `auto_config.py` for custom configurations
2. Use `python llm.py` for manual parameter tuning
3. Enable profiling with `python train_with_profiler.py`

### For Developers
1. Check out `PROFILER_README.md` for advanced profiling
2. Explore the codebase structure
3. Contribute improvements via pull requests

## 🎉 Success!

If you see training progress and a final model saved as `blueberry_model.pt`, you're all set!

**Happy training!** 🫐⚡

---

**Need help?** Open an issue on GitHub or check the main README.md for more details.
