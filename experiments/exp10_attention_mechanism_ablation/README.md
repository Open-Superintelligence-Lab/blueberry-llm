# Experiment 10: Attention Mechanism Ablation Study

A systematic ablation study comparing different attention mechanisms in isolation to understand their individual contributions to model performance.

## Overview

This experiment tests 11 different attention mechanisms in identical model architectures to isolate the impact of the attention computation method on:
- Training dynamics (loss curves, convergence speed)
- Final model quality (validation loss, perplexity)
- Computational efficiency (training time, memory usage)
- Parameter efficiency (parameter count vs performance)

## Research Questions

1. **How do different attention mechanisms compare in terms of model quality?**
   - Which attention type achieves the best validation loss and perplexity?
   - Are there clear winners or is performance mechanism-dependent?

2. **What is the efficiency/quality tradeoff?**
   - Do efficient mechanisms (MQA, GQA, Sliding Window) sacrifice much quality?
   - Is the extra computation of MHA worth it over simpler alternatives?

3. **Can we identify optimal attention choices for different use cases?**
   - Memory-constrained: MQA, GQA
   - Speed-constrained: Linear, Sliding Window
   - Quality-first: MHA, MLA

4. **Is attention even necessary?**
   - How much does the Identity mechanism (no attention) degrade performance?
   - Can this inform layer-wise attention scheduling?

## Attention Mechanisms Tested

### 1. Multi-Head Attention (MHA) - Baseline
**Standard Transformer attention**
- Full O(n²) attention computation
- Each head has independent Q, K, V projections
- Most expressive but most expensive
- **Use case**: Quality-first applications, reference baseline

### 2. Multi-Head Latent Attention (MLA)
**Low-rank KV compression (DeepSeek-V2/V3)**
- Compresses K and V through low-rank bottleneck
- Reduces KV cache size significantly
- Maintains most of MHA's expressiveness
- **Use case**: Long-context serving with limited memory

### 3. Grouped Query Attention (GQA)
**Shares KV across query head groups**
- Middle ground between MHA and MQA
- Tested with 2 KV heads (2 query heads per KV head)
- Reduces parameters and KV cache
- **Use case**: Practical efficiency without major quality loss

### 4. Multi-Query Attention (MQA)
**Single shared KV for all query heads**
- Extreme parameter reduction
- Fastest inference with minimal KV cache
- May lose some representational capacity
- **Use case**: Maximum efficiency, edge deployment

### 5. Sliding Window Attention
**Local attention within fixed window**
- Each token attends to w previous tokens
- O(n·w) complexity instead of O(n²)
- Tested with windows of 512 and 256 tokens
- **Use case**: Long sequences, streaming applications

### 6. Sparse Attention (DeepSeek-style)
**Learned adaptive sparsity**
- Uses indexer network to select top-k tokens
- Content-based rather than position-based sparsity
- Tested with k=512 and k=256
- **Use case**: Long context with adaptive selection

### 7. Linear Attention (Gated DeltaNet via FLA)
**O(n) complexity through linear attention**
- Recurrent formulation avoids quadratic cost
- Uses kernel tricks to approximate attention
- From Flash Linear Attention library
- **Use case**: Very long sequences, online inference

### 8. Identity (No Attention)
**Simple projection without attention**
- Tests if attention is necessary at all
- Provides lower bound on performance
- **Use case**: Understanding attention's contribution

## Model Architecture

All experiments use identical model architecture except for the attention mechanism:

```
- Layers: 4
- Model Dimension: 128
- Attention Heads: 4
- Feed-forward Dimension: 512
- Vocabulary Size: 32,000 (GPT-2 tokenizer)
- Max Sequence Length: 1024
- Dropout: 0.1
```

**Training Configuration:**
- Dataset: WikiText-2
- Training Steps: 1,000
- Batch Size: 4
- Learning Rate: 3e-4 with cosine schedule
- Warmup Steps: 100
- Optimizer: AdamW (β₁=0.9, β₂=0.95, weight_decay=0.01)

## Usage

### Run Single Attention Mechanism

```bash
# Test specific mechanism
python experiments/exp10_attention_mechanism_ablation/run_experiment.py --mechanism mha

# Available mechanisms:
# mha, mla, gqa_2, gqa_1, mqa, sliding_512, sliding_256, 
# sparse_512, sparse_256, linear, identity
```

### Run Comprehensive Ablation

```bash
# Test all mechanisms (default)
python experiments/exp10_attention_mechanism_ablation/run_experiment.py --mechanism all

# Custom training steps
python experiments/exp10_attention_mechanism_ablation/run_experiment.py --mechanism all --steps 2000
```

### Run Quick Test

```bash
# Quick test with fewer steps
python experiments/exp10_attention_mechanism_ablation/run_experiment.py --mechanism mha --steps 100
```

## Expected Results

Results will be saved to `experiments/exp10_attention_mechanism_ablation/results/`:

```
results/
├── comprehensive_results.json      # Summary of all experiments
├── comparison.png                  # Visualization comparing mechanisms
├── mha_baseline/
│   ├── config.json
│   └── metrics.json
├── mla/
│   ├── config.json
│   └── metrics.json
...
```

### Metrics Tracked

For each mechanism:
- **Training loss curve**: Loss at each logging step
- **Validation loss**: Evaluated every 100 steps
- **Perplexity**: exp(validation_loss)
- **Training time**: Total wall-clock time
- **Parameter count**: Total, attention, feedforward breakdowns

## Analysis Tools

### View Results Summary

```python
import json

with open('experiments/exp10_attention_mechanism_ablation/results/comprehensive_results.json') as f:
    results = json.load(f)

# Print ranking
sorted_results = sorted(results.items(), key=lambda x: x[1]['final_loss'])
for rank, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{rank}. {name}: Loss {metrics['final_loss']:.4f}, PPL {metrics['final_perplexity']:.2f}")
```

### Plot Training Curves

```python
import json
import matplotlib.pyplot as plt

mechanisms = ['mha', 'gqa_2', 'mqa', 'linear']
for mech in mechanisms:
    with open(f'experiments/exp10_attention_mechanism_ablation/results/{mech}/metrics.json') as f:
        data = json.load(f)
    plt.plot(data['train_steps'], data['train_loss'], label=mech)

plt.xlabel('Training Steps')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Training Loss Comparison')
plt.show()
```

## Hypotheses

### Expected Performance Ranking (Quality)
1. **MHA** - Full attention baseline
2. **MLA** - Compressed but expressive
3. **GQA** - Good balance
4. **Sliding Window** - Local context sufficient for small model
5. **MQA** - Reduced capacity
6. **Sparse** - May need more training to learn indexer
7. **Linear** - Different inductive bias
8. **Identity** - Worst, proves attention is valuable

### Expected Efficiency Ranking (Speed)
1. **Identity** - No computation
2. **Linear** - O(n) complexity
3. **MQA** - Minimal KV operations
4. **GQA** - Reduced KV operations
5. **Sliding Window** - Local computation
6. **Sparse** - Indexer overhead
7. **MLA** - Compression/decompression overhead
8. **MHA** - Full O(n²) attention

## Insights to Gain

1. **Quality-Efficiency Frontier**
   - Where is the sweet spot between performance and efficiency?
   - Is GQA truly the best practical choice?

2. **Architectural Choices**
   - Can we use different mechanisms at different layers?
   - Early layers: Local (sliding window)
   - Middle layers: Efficient (GQA, MQA)
   - Late layers: Expressive (MHA, MLA)

3. **Scaling Implications**
   - Which mechanisms will scale better to larger models?
   - Which are most sample-efficient?

4. **Attention Necessity**
   - How much does attention contribute vs feedforward layers?
   - Can we skip attention at some layers?

## Extensions

### Future Experiments

1. **Scale Up**: Test on larger models (12-24 layers, 512-1024 dim)
2. **Different Datasets**: Test on code, math, multilingual data
3. **Hybrid Architectures**: Mix mechanisms across layers
4. **Hyperparameter Tuning**: Each mechanism may prefer different LR
5. **Long Context**: Test at 4K, 8K, 16K sequence lengths
6. **Inference Benchmarks**: Measure actual throughput and memory

### Variations to Try

```python
# Test GQA with different KV head counts
configs = [
    get_gqa_config(n_kv_heads=1),   # Same as MQA
    get_gqa_config(n_kv_heads=2),   # Baseline
    get_gqa_config(n_kv_heads=4),   # Same as MHA
]

# Test sliding window sizes
configs = [
    get_sliding_window_config(window_size=128),
    get_sliding_window_config(window_size=256),
    get_sliding_window_config(window_size=512),
    get_sliding_window_config(window_size=1024),  # Full context
]

# Test sparse top-k values
configs = [
    get_sparse_config(sparse_top_k=128),
    get_sparse_config(sparse_top_k=256),
    get_sparse_config(sparse_top_k=512),
]
```

## Dependencies

```bash
pip install torch transformers datasets torchtune matplotlib
pip install fla-flash-linear-attention  # For linear attention
```

## File Structure

```
exp10_attention_mechanism_ablation/
├── __init__.py                    # Experiment description
├── attention_mechanisms.py        # All attention implementations
├── models.py                      # Model wrapper and blocks
├── config.py                      # Configuration classes
├── run_experiment.py              # Main training script
├── README.md                      # This file
└── results/                       # Generated results
    ├── comprehensive_results.json
    ├── comparison.png
    └── {mechanism_name}/
        ├── config.json
        └── metrics.json
```

## Related Work

- **MHA**: Vaswani et al., "Attention Is All You Need" (2017)
- **MQA**: Shazeer, "Fast Transformer Decoding" (2019)
- **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer" (2023)
- **MLA**: DeepSeek-V2/V3 Technical Reports (2024-2025)
- **Linear Attention**: Katharopoulos et al., "Transformers are RNNs" (2020)
- **DeltaNet**: FLA library implementation
- **Sparse Attention**: Child et al., "Generating Long Sequences with Sparse Transformers" (2019)
- **DeepSeek Sparse**: DeepSeek-V3 Technical Report (2024)

## Citation

```bibtex
@experiment{blueberry-exp10-2025,
  title={Attention Mechanism Ablation Study},
  author={Blueberry-LLM Experiments},
  year={2025},
  note={Systematic comparison of 11 attention mechanisms}
}
```

## Contributing

To add a new attention mechanism:

1. Implement in `attention_mechanisms.py` following the common interface
2. Add config function in `config.py`
3. Add to `ALL_CONFIGS` dictionary
4. Update README with mechanism description

Example:
```python
# In attention_mechanisms.py
class MyNewAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, **kwargs):
        super().__init__()
        # Your implementation
    
    def forward(self, x):
        # Returns: [batch_size, seq_len, d_model]
        pass

# In config.py
def get_mynew_config() -> AttentionAblationConfig:
    return AttentionAblationConfig(
        attention_type="mynew",
        experiment_name="mynew_attention"
    )

# Add to ALL_CONFIGS
ALL_CONFIGS["mynew"] = get_mynew_config
```

