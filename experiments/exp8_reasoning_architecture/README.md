# Experiment 8: Reasoning Architecture

Building reasoning capabilities on top of the winning architecture from Experiment 7, with optional **Recursive Reasoning** using Adaptive Compute Time (ACT).

## Two Modes

### 1. Baseline Mode
Uses **Hybrid Sparse 17%** directly from Experiment 7:

```python
# Configuration
hidden_size: 768
num_layers: 12
attention_layers: [5, 11]  # 17% attention (mid and near-end)
learning_rate: 0.002
batch_size: 48
seq_len: 1024
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

## Why Exp7's Architecture?

From Experiment 7 results:
- **Best validation loss**: 4.055 (best of 13 architectures tested)
- **27% better** than pure Transformer (5.146)
- **8% better** than pure DeltaNet (4.396)
- **Throughput**: 118K tokens/sec
- **Strategic placement**: Attention at layers 5 (mid) and 11 (near-end)

## Recursive Reasoning Features

Based on Tiny Recursive Models (TRM):
- **Hierarchical Cycles**: Two-level reasoning (H and L)
- **Adaptive Compute Time**: Q-learning based halting
- **Carry State**: Maintains reasoning context across iterations
- **Iterative Refinement**: Multiple passes for complex reasoning

## Architecture Details

### Layer Configuration
- **DeltaNet layers** (10): [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
  - Efficient O(n) processing
  - Strong inductive bias for sequential patterns
  
- **Attention layers** (2): [5, 11]
  - Layer 5: Captures intermediate representations
  - Layer 11: Refines high-level features before prediction

### Model Stats
- **Parameters**: ~302M (GatedDeltaNet with hybrid attention)
- **Context length**: 1024 tokens
- **Vocabulary**: ~50K tokens

## Usage

### Training Modes

#### 1. Train Baseline Only (Default)
```bash
# Default: baseline with exp7 architecture
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

1. **Baseline**: Establish baseline with exp7 winner (Hybrid Sparse 17%)
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

## Interpreting Results

### What to Look For

**Baseline**: Clean exp7 winner performance
- Val loss should match or be close to exp7 results (~4.055)
- Steady learning curve
- No reasoning overhead

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
- ✅ Baseline performance benchmark
- ✅ Recursive reasoning infrastructure
- ✅ ACT-based adaptive depth
- ✅ Comparison framework

Future enhancements:
- Chain-of-thought prompting integration
- Multi-step reasoning tasks
- Verification and self-correction loops
- Task-specific reasoning objectives

## References

- Based on [Experiment 7](../exp7_hybrid_deltanet_ablation/README.md) findings
- Uses [FLA (Flash Linear Attention)](https://github.com/sustcsonglin/flash-linear-attention) library
- GatedDeltaNet architecture with hybrid attention support
- Recursive reasoning inspired by [Tiny Recursive Models](https://github.com/google-deepmind/tiny-recursive-models)

