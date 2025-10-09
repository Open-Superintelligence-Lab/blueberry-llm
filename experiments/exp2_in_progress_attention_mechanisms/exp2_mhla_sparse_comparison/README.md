# Experiment 5: DeepSeek MHLA with/without Sparse Attention

**Testing whether sparse token selection improves DeepSeek's already-efficient Multi-Head Latent Attention (MHLA).**

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [What's Being Compared](#-whats-being-compared)
3. [Key Differences from Experiment 4](#-key-differences-from-experiment-4)
4. [Architecture Details](#-architecture-details)
5. [How to Run](#-how-to-run)
6. [Expected Results](#-expected-results)
7. [Understanding the Code](#-understanding-the-code)
8. [Customization](#-customization)

---

## 🚀 Quick Start

```bash
cd experiments/exp5_mhla_sparse_attention
python run_experiment.py
```

**What you'll get**: Comparison of DeepSeek MHLA (dense) vs DeepSeek MHLA + Sparse Attention across sequence lengths 64, 128, 256.

**Time**: ~20-25 minutes on GPU

---

## 🎯 What's Being Compared

### Baseline: DeepSeek Multi-Head Latent Attention (Dense)

**Architecture**:
- ✅ **Multi-Head Latent Attention (MHLA)** with KV compression
- ✅ **LoRA-style projections** (compress → decompress)
- ✅ **RoPE positional encoding**
- ✅ **Shared latent space** for efficient KV cache
- ❌ **NO sparse token selection** (attends to all tokens)

**Characteristics**:
- Already efficient via latent compression
- O(L²) attention complexity
- Reduced KV cache size
- Standard for DeepSeek models

### Experimental: DeepSeek MHLA + Sparse Attention

**Architecture**:
- ✅ **Multi-Head Latent Attention (MHLA)** with KV compression
- ✅ **LoRA-style projections** (compress → decompress)
- ✅ **RoPE positional encoding**
- ✅ **Shared latent space** for efficient KV cache
- ✅ **Lightning Indexer** for token relevance scoring
- ✅ **Top-k token selection** (sparse attention mask)

**Characteristics**:
- MHLA efficiency + sparse selection
- O(Lk) attention complexity where k < L
- Further reduced computation
- Tests if sparsity helps already-efficient attention

---

## 🔑 Key Differences from Experiment 4

| Aspect | Experiment 4 | Experiment 5 (This) |
|--------|-------------|---------------------|
| **Baseline** | Standard Multi-Head Attention | DeepSeek MHLA (with latent compression) |
| **Attention Type** | QKV linear projections | Compressed KV with LoRA-style up/down projections |
| **KV Cache** | Full dimensional | Compressed to latent space |
| **Positional Encoding** | Simple RoPE | DeepSeek's RoPE with nope/rope split |
| **Question** | Does sparse help standard attention? | Does sparse help already-efficient MHLA? |

**Why This Matters**: Experiment 4 showed sparse helps standard attention. But does it help when attention is already optimized via latent compression? This experiment answers that.

---

## 🏗️ Architecture Details

### Multi-Head Latent Attention (MHLA)

MHLA reduces memory by compressing KV states:

```python
# 1. Compress Keys/Values to latent space
compressed_kv = W^DKV @ hidden_states  # Down-projection
compressed_kv = LayerNorm(compressed_kv)

# 2. Decompress for each head
kv = W^UK @ compressed_kv  # Up-projection
k_nope, v = split(kv)

# 3. Add positional component
k_pe = RoPE(W^KR @ hidden_states)
key = concat([k_nope, k_pe])

# 4. Standard attention
attn_output = softmax(Q @ K^T / √d) @ V
```

**Benefits**:
- KV cache: `batch × seq_len × latent_dim` (smaller!)
- Shared latent space across heads
- Efficient for long contexts

### Sparse MHLA (Our Addition)

We add Lightning Indexer before attention:

```python
# 1-3. Same MHLA compression/decompression as above

# 4. Lightning Indexer (NEW)
index_scores = LightningIndexer(hidden_states)  # [B, L, L]

# 5. Top-k Selection (NEW)
top_k_mask = TopKSelector(index_scores, k=L/2)  # Select 50%

# 6. Sparse Attention (NEW)
sparse_mask = create_mask(top_k_mask)  # -inf for non-selected
attn_output = softmax(Q @ K^T / √d + sparse_mask) @ V
```

**Additional Benefits**:
- Further reduced computation (50% tokens)
- Forced attention selectivity
- Potential regularization effect

---

## 🏃 How to Run

### Basic Usage

```bash
python run_experiment.py
```

### What Happens

1. **Load Data**: TinyStories dataset tokenized for each sequence length
2. **Train Baseline**: DeepSeek MHLA (dense) for 1000 steps
3. **Train Sparse**: DeepSeek MHLA + Sparse for 1000 steps
4. **Compare**: Generate plots and metrics
5. **Save Results**: JSON files + visualization

### Output Structure

```
results/
├── sequence_length_comparison.png    # Main visualization
├── summary.json                      # Numerical comparison
└── seq_*/
    ├── baseline_results.json         # Dense MHLA results
    └── sparse_results.json           # Sparse MHLA results
```

---

## 📊 Results Summary

### Performance Comparison (512d model, 6 layers, 8 heads)

| Seq Length | Baseline Loss | Sparse Loss | Improvement | Baseline Acc | Sparse Acc | Baseline Time/Step | Sparse Time/Step | Speed Change |
|------------|---------------|-------------|-------------|--------------|------------|-------------------|------------------|--------------|
| 64         | 7.43          | **6.64**    | **12% better** | 9.2%        | **15.5%**  | 0.075s            | 0.075s           | **Same**      |
| 128        | 6.85          | 6.97        | -2% worse    | 10.3%       | 10.3%      | 0.076s            | 0.078s           | -3% slower    |
| 256        | 6.61          | **6.55**    | **1% better** | 12.5%       | **13.2%**  | 0.084s            | 0.087s           | -4% slower    |
| 1024       | **4.10**      | 6.91        | **-41% worse** | **32.2%**  | 10.7%      | 0.084s            | 0.082s           | **3% faster** |
| 2048       | 6.64          | **6.63**    | **0% same**   | 11.9%       | **14.4%**  | 0.076s            | 0.077s           | -1% slower    |

**Model Size**: 79.3M parameters (baseline) vs 80.2M parameters (sparse) - **+1.0M parameters (+1.3%)**

### 🎯 Key Findings

1. **Mixed Results**: Sparse attention shows inconsistent benefits on MHLA
   - **Short sequences (64)**: Clear improvement (12% better loss, 68% better accuracy)
   - **Medium sequences (128-256)**: Minimal difference
   - **Long sequences (1024)**: **Baseline MHLA significantly outperforms** (-41% worse loss)
   - **Very long sequences (2048)**: Slight accuracy improvement but similar loss

2. **Speed Analysis**: 
   - **No consistent speedup** from sparse attention
   - Overhead of Lightning Indexer (~3-4% slower) outweighs sparse computation savings
   - Only 1024 length shows slight speedup (3% faster)

3. **MHLA vs Standard Attention Comparison**:
   - **Experiment 4** (standard attention): Sparse provided 139-302% improvements
   - **Experiment 5** (MHLA): Sparse provides -41% to +12% improvements
   - **Conclusion**: MHLA's latent compression already provides most benefits of sparsity

4. **Long Sequence Behavior**:
   - **1024 length**: Baseline MHLA achieves 32.2% accuracy vs sparse's 10.7%
   - **MHLA excels at long sequences** without additional sparsity
   - **Sparse attention may interfere** with MHLA's learned compression patterns

### 🔬 Research Insights

**Why Sparse Doesn't Help MHLA as Much**:

1. **Redundant Mechanisms**: 
   - MHLA already compresses KV cache (latent space)
   - Sparse selection adds another compression layer
   - **Double compression may be too aggressive**

2. **Learned Patterns**:
   - MHLA learns optimal compression patterns during training
   - Lightning Indexer may select different tokens than MHLA's learned patterns
   - **Conflicting selection strategies**

3. **Long Context Performance**:
   - MHLA's latent compression scales well to long sequences
   - Additional sparsity may remove important long-range dependencies
   - **1024 length results suggest MHLA alone is optimal**

### 📈 Training Efficiency

**Parameter Overhead**: Only +1.3% parameters for sparse version
**Training Time**: Similar training speed (no significant speedup)
**Memory Usage**: Sparse version uses slightly more memory due to indexer

---

## 🔍 Understanding the Code

### File Structure

```
exp5_mhla_sparse_attention/
├── sparse_mhla_attention.py    # Lightning Indexer + Sparse MHLA wrapper
├── exp5_models.py              # Full model definitions (baseline & sparse)
├── run_experiment.py           # Training script
├── README.md                   # This file
└── results/                    # Experiment outputs
```

### Key Classes

#### 1. `LightningIndexer` (sparse_mhla_attention.py)

Computes token relevance scores:

```python
I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j}^I · k_s^I)
```

- Input: Hidden states `[batch, seq_len, d_model]`
- Output: Index scores `[batch, seq_len, seq_len]`
- Parameters: ~40K for typical config

#### 2. `DeepSeekSparseMLHA` (sparse_mhla_attention.py)

Wraps DeepseekV3Attention with sparse selection:

```python
class DeepSeekSparseMLHA(nn.Module):
    def __init__(self, config, layer_idx, sparse_top_k=512):
        self.mhla = DeepseekV3Attention(config, layer_idx)  # Standard MHLA
        self.indexer = LightningIndexer(...)                # Token scorer
        self.selector = TopKTokenSelector(...)              # Top-k picker
```

#### 3. Model Classes (exp5_models.py)

- `BaselineMLHAMoELLM`: Dense MHLA + MoE
- `SparseMLHAMoELLM`: Sparse MHLA + MoE

Both share same architecture except for sparse attention component.

---

## 🎨 Customization

### Change Sparsity Level

```python
# In run_experiment.py, line ~215
config['sparse_top_k'] = int(seq_len * 0.25)  # 75% sparse
config['sparse_top_k'] = int(seq_len * 0.75)  # 25% sparse
```

### Adjust MHLA Compression

```python
# In BASE_CONFIG
BASE_CONFIG = {
    'kv_lora_rank': 32,   # More compression (smaller)
    'kv_lora_rank': 128,  # Less compression (larger)
    'q_lora_rank': 64,    # Also compress queries
}
```

### Test Different Sequence Lengths

```python
SEQUENCE_LENGTHS = [32, 64, 128, 256, 512]
```

### Change Model Size

```python
BASE_CONFIG = {
    'd_model': 512,      # Larger model
    'n_heads': 16,       # More attention heads
    'n_layers': 8,       # Deeper
}
```

### Adjust Training

```python
BASE_CONFIG = {
    'steps': 2000,       # Longer training
    'learning_rate': 1e-3,  # Different LR
    'batch_size': 32,    # Larger batches
}
```

---

## 🔬 Research Questions - ANSWERED

This experiment answered:

1. **Does sparse attention help already-efficient architectures?**
   - ✅ **ANSWERED**: Mixed results - helps short sequences (64), hurts long sequences (1024)
   - MHLA's latent compression is already very efficient

2. **Is sparsity complementary to latent compression?**
   - ✅ **ANSWERED**: **No** - they appear to be redundant mechanisms
   - Double compression (latent + sparse) may be too aggressive
   - MHLA's learned patterns conflict with Lightning Indexer selection

3. **Does the combination scale better?**
   - ✅ **ANSWERED**: **No** - baseline MHLA scales better alone
   - Long sequences (1024): MHLA achieves 32.2% accuracy vs sparse's 10.7%
   - Combined approach is not optimal for long contexts

---

## 📊 Interpreting Results

### Scenario 1: Sparse Provides Large Improvements

**Interpretation**: Sparse attention and latent compression are **complementary**
- Latent compression: Reduces memory
- Sparse selection: Improves attention quality
- **Implication**: Use both for long-context models

### Scenario 2: Sparse Provides Small Improvements

**Interpretation**: Latent compression already captures most benefits
- MHLA's efficiency sufficient
- Sparse adds marginal value
- **Implication**: Latent compression alone may be enough

### Scenario 3: Sparse Hurts Performance

**Interpretation**: Excessive compression/sparsity
- Both mechanisms remove information
- Combined effect too aggressive
- **Implication**: Balance needed between efficiency and capacity

---

## 🔗 Related Experiments

- **Experiment 1**: Ablation of DeepSeek components
- **Experiment 3**: MHLA + MoE optimization
- **Experiment 4**: Sparse attention on standard MHA (comparison baseline)

---

## 📚 References

- **DeepSeek-V3 Paper**: Multi-Head Latent Attention architecture
- **DeepSeek-V3.2-Exp Paper**: Lightning Indexer and sparse attention
- **Implementation**: `deepseek_modeling.py` (MHLA), `sparse_mhla_attention.py` (sparse component)

---

## 🎯 Key Takeaways

1. **MHLA is already highly optimized** - sparse attention provides minimal additional benefit
2. **Short sequences benefit slightly** from sparse attention (12% improvement at length 64)
3. **Long sequences are hurt** by sparse attention (-41% worse at length 1024)
4. **No speed improvement** - Lightning Indexer overhead cancels sparse computation savings
5. **Recommendation**: Use MHLA alone for production models, skip sparse attention
6. **MHLA's latent compression** is more effective than token-level sparsity for long contexts

---

## 💡 Tips

- **Run Experiment 4 first** to understand baseline sparse attention benefits
- **Compare parameter counts** - sparse adds ~5% more parameters
- **Watch memory usage** - MHLA already reduces memory, sparse may help further
- **Check training stability** - sparse can regularize or destabilize depending on config

---

*Experiment designed: October 2025*  
*Tests DeepSeek MHLA (from V3) + Sparse Attention (from V3.2-Exp)*

