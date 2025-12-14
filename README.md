# Looking for a name for this LLM

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
pip install -r requirements.txt

python train_moe.py
```

## Current baseline: 3B MoE

This project implements a **~3 billion parameter Mixture of Experts (MoE)** transformer model with the following specifications:

### Architecture
- **Model Dimension** (`d_model`): 1536
- **Attention Heads** (`n_heads`): 12
- **Layers** (`n_layers`): 26
- **Feed-Forward Dimension** (`d_ff`): 4096 (~2.67x d_model)
- **Head Dimension** (`d_k`): 128 (d_model / n_heads)
- **Dropout**: 0.1

### Multi-Latent Attention (MLA) - Optional
- **MLA Enabled**: False (by default)
- **QK RoPE Dimension**: 32
- **QK NoPE Dimension**: 128
- **KV LoRA Rank**: 64
- **Value Dimension**: 128

### MoE Configuration
- **Number of Experts**: 8
- **Top-K Experts per Token**: 2
- **Load Balancing Weight**: 0.01

### Training Parameters
- **Batch Size**: 8
- **Gradient Accumulation Steps**: 12
- **Effective Batch Size**: 96 (8 × 12)
- **Max Training Steps**: 10,000
- **Warmup Ratio**: 5%

### Optimizers
- **Muon Optimizer**:
  - Learning Rate: 0.02
  - Momentum: 0.95
- **AdamW Optimizer**:
  - Learning Rate: 0.003
  - Weight Decay: 0.2

### Data Configuration
- **Max Sequence Length**: 512 tokens
- **Number of Documents**: 2,000
- **Max Tokens**: 500,000

### Regularization & Training Settings
- **Weight Decay**: 0.2
- **Gradient Clipping**: 1.0
- **Mixed Precision (AMP)**: Enabled
- **Evaluation Frequency**: Every 10 steps
- **Evaluation Steps**: 100
- **Logging Milestones**: 2,000 / 5,000 / 10,000 steps

## Getting Started

1. **Fork this repository** - Click the "Fork" button at the top right of this page to create your own copy
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/blueberry-llm.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Read `CONTRIBUTING.md` for contribution guidelines
5. Create your own experiment and merge it
6. Explore the `experiments/` folder for ongoing research and inspiration
7. Once you finish with your research, create a pull request to merge it back to this repo

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project