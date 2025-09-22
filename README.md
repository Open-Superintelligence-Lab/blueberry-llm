# Blueberry LLM 🫐

A Mixture of Experts (MoE) language model implementation.

**Goal: Make LLM training accessible to everyone** - anyone, regardless of technical background, can train SOTA language models on any GPU setup (1-1M GPUs) with automatic best settings and hardware optimization to create state of the art LLM.

## Quick Start

```bash
git clone https://github.com/Open-Superintelligence-Lab/blueberry-llm
cd blueberry-llm
chmod +x setup.sh
./setup.sh
python train.py
```

Try it on [Google Colab](https://colab.research.google.com/drive/1UE82keuNStPPaeCF50zSgXVHWiywo_pm?usp=sharing)

```bash
# Quick test with 500 steps (fast validation)
# python core/train.py --config dev --max-steps 500
# I think this doesn't work, line above

# Use Megatron for distributed training (optional)
python core/train_auto.py --use-megatron
```

Integrates **NVIDIA's Megatron-LM** for massive GPU cluster training, native PyTorch used for smaller setups. Manual choice still needed, more testing needed.

## 🖥️ Autotuned GPU Setups

Blueberry LLM automatically detects your hardware and optimizes configuration for maximum performance:

| GPU | Memory | Status | Auto-Optimization | Notes |
|-----|--------|--------|------------------|-------|
| **Tesla T4** (Google Colab) | 16GB | ✅ **Fully Supported** | ✅ Optimized | Max memory utilization (~13-14GB) |

---

This is an **open research project** - we encourage everyone to fork the project, run experiments, and submit pull requests with improvements.

## 📁 Project Structure

```
blueberry-llm/
├── 📁 core/                    # Main functionality
│   ├── train.py               # Main training script
│   ├── train_auto.py          # Auto-configuration training
│   ├── inference.py           # Model inference
│   └── auto_config.py         # Auto-configuration logic
├── 📁 models/                 # Neural network components
├── 📁 data/                   # Data pipeline
├── 📁 optimizers/             # Advanced optimizers
├── 📁 training/               # Training infrastructure
├── 📁 ops/                    # GPU-adaptive operations
├── 📁 system/                 # Hardware detection
├── 📁 configs/                # Configuration management
├── 📁 tests/                  # Testing and examples
├── 📁 docs/                   # Documentation
└── 📁 legacy/                 # Legacy files for reference
```

## 🚀 Usage

### Training
```bash
# Auto-configured training (recommended)
python train.py

# Quick test with 500 steps (fast validation)
python core/train.py --config dev --max-steps 500

# Manual configuration
python core/train.py --config dev
python core/train.py --d-model 768 --n-layers 12
```

### Inference
```bash
# Generate text from trained model
python inference.py "Your prompt here"

# Interactive mode
python inference.py --interactive
```

### Testing
```bash
# Run GPU-adaptive system tests
python test.py

# Run integration examples
python tests/example_integration.py
```

## Research Questions

- Can we achieve better parameter efficiency with sparse expert activation?
- Can we improve expert routing? How does load balancing affect model performance and convergence?

## Engineering Questions
- Should we make it auto detect number and type of GPUs?

## Future Project

Test with [Token Order Prediction](https://github.com/zaydzuhri/token-order-prediction) - "Predicting the Order of Upcoming Tokens Improves Language Modeling"

## Contributing

We welcome contributions! Fork the repo, experiment with different architectures, and submit PRs with your findings.

## Vision

Any company or person (even with no technical experience) should be able to download this repository and run it on their GPU setup - from 1 GPU to 1 million GPUs. The system will be able to automatically detect your hardware configuration, tune hyperparameters for optimal performance, and run the best possible training with or without manual configuration from your side.
