# Experiment 8: Reasoning Architecture

Building reasoning capabilities on top of the winning architecture from Experiment 7.

## Base Architecture

Uses **Hybrid Sparse 17%** - the best performing architecture from Experiment 7:

```python
# Configuration
hidden_size: 768
num_layers: 12
attention_layers: [5, 11]  # 17% attention (mid and near-end)
learning_rate: 0.002
batch_size: 48
seq_len: 1024
```

### Why This Architecture?

From Experiment 7 results:
- **Best validation loss**: 4.055 (best of 13 architectures tested)
- **27% better** than pure Transformer (5.146)
- **8% better** than pure DeltaNet (4.396)
- **Throughput**: 118K tokens/sec
- **Strategic placement**: Attention at layers 5 (mid) and 11 (near-end)

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

### Basic Training

```bash
# Default configuration (1000 steps)
python run_experiment.py

# Extended training (5000 steps)
python run_experiment.py --experiment extended
```

### Resume & Extend

```bash
# Resume from checkpoint
python run_experiment.py --resume checkpoints/best_model.pt

# Resume and extend to 10000 steps
python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 10000
```

## Experiment Goals

1. **Baseline**: Establish baseline performance with exp7 winner architecture
2. **Reasoning Extensions**: Add reasoning capabilities (to be implemented)
   - Chain-of-thought mechanisms
   - Multi-step reasoning
   - Verification and correction loops

## Results

Results will be saved to:
- `results/training_results.json` - Training metrics and configuration
- `results/training_curves.png` - Visualization of training progress
- `checkpoints/best_model.pt` - Best model checkpoint
- `checkpoints/final_model.pt` - Final model checkpoint

## Next Steps

This experiment establishes the foundation. Future work will add:
- Reasoning-specific training objectives
- Chain-of-thought prompting mechanisms
- Multi-step reasoning evaluation
- Verification and self-correction capabilities

## References

- Based on [Experiment 7](../exp7_hybrid_deltanet_ablation/README.md) findings
- Uses [FLA (Flash Linear Attention)](https://github.com/sustcsonglin/flash-linear-attention) library
- GatedDeltaNet architecture with hybrid attention support

