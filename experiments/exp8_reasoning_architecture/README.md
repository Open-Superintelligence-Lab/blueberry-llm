# Experiment 8: Reasoning Architecture

Building reasoning capabilities using **MoE (Mixture of Experts)** architecture, with optional **Recursive Reasoning** using Adaptive Compute Time (ACT).

## Two Modes

### 1. Baseline Mode
Uses **MoE architecture** with sparse expert activation (optimized for RTX 4090):

```python
# Configuration (4090-optimized)
hidden_size: 512        # Reduced for 24GB VRAM
num_layers: 8           # Reduced from 12
num_experts: 4          # Experts per MoE layer (reduced from 8)
expert_top_k: 2         # Active experts per token
learning_rate: 0.002
batch_size: 16          # Reduced from 48
seq_len: 512            # Reduced from 1024
```

### 2. Recursive Reasoning Mode  
Wraps the baseline with **hierarchical reasoning cycles** and **ACT halting**:

```python
# Additional Recursive Config
H_cycles: 3          # High-level reasoning iterations
L_cycles: 3          # Low-level iterations per H cycle
halt_max_steps: 5    # Maximum reasoning steps
use_act: True        # Adaptive Compute Time enabled
```

## Why MoE Architecture?

Benefits of Mixture of Experts:
- **Increased model capacity** without proportional compute increase
- **Specialized experts** for different input patterns
- **Sparse activation** - only top-k experts active per token
- **Scalable** - can add more experts for more capacity
- **Efficient** - constant compute per token regardless of expert count

## Recursive Reasoning Features

Based on Tiny Recursive Models (TRM):
- **Hierarchical Cycles**: Two-level reasoning (H and L)
- **Adaptive Compute Time**: Q-learning based halting
- **Carry State**: Maintains reasoning context across iterations
- **Iterative Refinement**: Multiple passes for complex reasoning

## Architecture Details

### MoE Layer Configuration (4090)
- **Transformer layers** (8): All layers use MoE
  - Self-attention mechanism
  - MoE feed-forward with 4 experts
  - Top-2 expert routing per token
  - Load balancing for expert utilization

### Model Stats (4090 Configuration)
- **Parameters**: ~50-100M (reduced for 4090)
- **Active parameters per token**: ~25-50M (due to sparse activation)
- **Context length**: 512 tokens
- **Vocabulary**: ~50K tokens
- **Memory footprint**: ~8-12GB VRAM (fits comfortably on 24GB)

## Usage

### Training Modes

#### 1. Train Baseline Only (Default)
```bash
# Default: baseline with MoE architecture
python run_experiment.py
```

#### 2. Train Recursive Only
```bash
# Train with recursive reasoning
python run_experiment.py --model-type recursive
```

#### 3. Compare Both Models
```bash
# Train both baseline and recursive, then compare
python run_experiment.py --compare
```

### Advanced Options

```bash
# Extended training (5000 steps)
python run_experiment.py --experiment extended --compare

# Resume from checkpoint
python run_experiment.py --resume checkpoints_baseline/best_model.pt

# Resume and extend to 10000 steps
python run_experiment.py --resume checkpoints_baseline/best_model.pt --extend-steps 10000
```

### Comparison Analysis

After running with `--compare`, or manually:
```bash
python compare_baseline_vs_recursive.py
```

Generates:
- `results/comparison_plots.png` - Visual comparison charts
- `results/comparison_summary.json` - Detailed metrics

## Experiment Goals

1. **Baseline**: Establish baseline with MoE architecture
2. **Recursive Reasoning**: Test hierarchical reasoning with ACT
3. **Comparison**: Quantify improvement from recursive reasoning
4. **Future**: Foundation for advanced reasoning capabilities

## Results Structure

### Baseline Training
- `checkpoints_baseline/best_model.pt` - Best baseline checkpoint
- `results/training_results_baseline.json` - Baseline metrics
- `results/training_curves_baseline.png` - Baseline learning curves

### Recursive Training
- `checkpoints_recursive/best_model.pt` - Best recursive checkpoint
- `results/training_results_recursive.json` - Recursive metrics with ACT stats
- `results/training_curves_recursive.png` - Recursive learning curves

### Comparison (when using --compare)
- `results/comparison_plots.png` - Side-by-side visualizations
- `results/comparison_summary.json` - Detailed comparison metrics

## Recursive Reasoning Metrics

When training with recursive reasoning, additional metrics are tracked:

- **reasoning_steps**: Average number of reasoning iterations per example
- **halt_rate**: Percentage of sequences that halt before max steps
- **q_halt_mean**: Average Q-value for halting decision
- **q_continue_mean**: Average Q-value for continuing
- **act_penalty**: Regularization loss encouraging efficient halting

## MoE-Specific Metrics

The MoE model also tracks:
- **aux_loss**: Load balancing auxiliary loss
- **expert_utilization**: Which experts are being used
- **routing_entropy**: Diversity of expert selection

## Interpreting Results

### What to Look For

**Baseline**: Clean MoE performance
- Stable training with expert load balancing
- Steady learning curve
- Efficient sparse computation

**Recursive**: Iterative refinement benefits
- Potentially lower val loss (if reasoning helps)
- Higher compute cost (more FLOPs per forward pass)
- ACT metrics show adaptive behavior

**Comparison**:
- Loss improvement %: Positive = recursive is better
- Training time: Recursive should be slower due to cycles
- Parameters: Recursive adds projection layers + ACT head

## Next Steps

This experiment provides:
- ✅ MoE baseline performance benchmark
- ✅ Recursive reasoning infrastructure
- ✅ ACT-based adaptive depth
- ✅ Comparison framework

Future enhancements:
- Chain-of-thought prompting integration
- Multi-step reasoning tasks
- Verification and self-correction loops
- Task-specific reasoning objectives
- Expert specialization analysis

## References

- Uses custom MoE implementation from `/models/moe_llm.py`
- MoE architecture with sparse expert activation
- Recursive reasoning inspired by [Tiny Recursive Models](https://github.com/google-deepmind/tiny-recursive-models)

