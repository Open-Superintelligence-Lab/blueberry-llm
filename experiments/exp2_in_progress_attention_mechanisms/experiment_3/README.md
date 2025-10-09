# Experiment 3: Dynamic Sparsity and Adaptive Attention Patterns

**Investigating adaptive sparsity patterns that dynamically adjust based on sequence characteristics to optimize both pretraining speed and quality.**

---

## 📋 Table of Contents

1. [Research Question](#-research-question)
2. [Hypothesis](#-hypothesis)
3. [Experimental Design](#-experimental-design)
4. [Architecture Overview](#-architecture-overview)
5. [Implementation Details](#-implementation-details)
6. [Evaluation Metrics](#-evaluation-metrics)
7. [Expected Outcomes](#-expected-outcomes)
8. [Scientific Rigor](#-scientific-rigor)

---

## 🎯 Research Question

**How can we design adaptive sparsity patterns that dynamically adjust based on sequence characteristics to optimize both pretraining speed and quality in transformer models?**

This experiment addresses multiple research questions from `RESEARCH_QUESTIONS.md`:
- "What is the optimal k value for different sequence lengths?" → **Dynamic k based on sequence properties**
- "How does indexer performance scale with sequence length?" → **Adaptive indexer behavior**
- "How does scaling influence indexer accuracy and computational efficiency?" → **Scaling-aware attention patterns**

---

## 🔬 Hypothesis

**Hypothesis**: Adaptive sparsity patterns that adjust based on sequence characteristics (length, content complexity, attention entropy) will outperform fixed sparsity patterns in both training speed and model quality.

**Rationale**:
1. **Sequence Length Adaptation**: Longer sequences may benefit from different sparsity levels
2. **Content Complexity Adaptation**: Complex sequences may need more attention capacity
3. **Attention Entropy Adaptation**: High-entropy attention patterns may indicate need for more tokens
4. **Dynamic Efficiency**: Optimal k values should vary per sequence, not be fixed

---

## 🧪 Experimental Design

### Baseline Models
1. **Fixed Sparse (50%)**: Standard DeepSeek sparse attention with k = L/2
2. **Fixed Sparse (25%)**: Standard DeepSeek sparse attention with k = L/4  
3. **Fixed Sparse (75%)**: Standard DeepSeek sparse attention with k = 3L/4
4. **Dense Attention**: Full attention baseline

### Experimental Models
1. **Length-Adaptive**: k = f(L) where f is learned function of sequence length
2. **Complexity-Adaptive**: k = g(complexity) based on token diversity and perplexity
3. **Entropy-Adaptive**: k = h(entropy) based on attention pattern entropy
4. **Hybrid-Adaptive**: k = combined(length, complexity, entropy) with learned weights

### Training Protocol
- **Dataset**: TinyStories (consistent with previous experiments)
- **Sequence Lengths**: 64, 128, 256, 512, 1024, 2048
- **Training Steps**: 2000 per configuration (sufficient for convergence)
- **Batch Size**: 16 (balanced for memory and gradient stability)
- **Learning Rate**: 1e-3 (standard for transformer pretraining)
- **Random Seeds**: Fixed seeds (42, 123, 456) for reproducibility

---

## 🏗️ Architecture Overview

### Dynamic Sparsity Controller

```python
class DynamicSparsityController(nn.Module):
    def __init__(self, d_model, max_seq_len):
        # Sequence length predictor
        self.length_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: normalized sparsity level
        )
        
        # Content complexity analyzer
        self.complexity_analyzer = ContentComplexityAnalyzer(d_model)
        
        # Attention entropy estimator
        self.entropy_estimator = AttentionEntropyEstimator(d_model)
        
        # Adaptive k calculator
        self.k_calculator = AdaptiveKCalculator(max_seq_len)
    
    def forward(self, hidden_states, seq_len):
        # Extract sequence characteristics
        length_factor = self.length_predictor(hidden_states.mean(dim=1))
        complexity_factor = self.complexity_analyzer(hidden_states)
        entropy_factor = self.entropy_estimator(hidden_states)
        
        # Calculate adaptive k
        adaptive_k = self.k_calculator(
            seq_len, length_factor, complexity_factor, entropy_factor
        )
        
        return adaptive_k
```

### Content Complexity Analyzer

```python
class ContentComplexityAnalyzer(nn.Module):
    def __init__(self, d_model):
        # Token diversity analyzer
        self.diversity_proj = nn.Linear(d_model, 1)
        
        # Perplexity estimator
        self.perplexity_estimator = PerplexityEstimator(d_model)
        
    def forward(self, hidden_states):
        # Token diversity (variance in embeddings)
        diversity = torch.var(hidden_states, dim=1).mean(dim=-1)
        
        # Perplexity estimation
        perplexity = self.perplexity_estimator(hidden_states)
        
        # Combine complexity signals
        complexity = torch.sigmoid(
            self.diversity_proj(diversity.unsqueeze(-1)) + 
            perplexity
        )
        
        return complexity
```

### Attention Entropy Estimator

```python
class AttentionEntropyEstimator(nn.Module):
    def __init__(self, d_model):
        self.entropy_proj = nn.Linear(d_model, 1)
        
    def forward(self, hidden_states):
        # Estimate attention entropy without computing full attention
        # Use variance in hidden states as proxy for attention entropy
        attention_proxy = torch.softmax(
            self.entropy_proj(hidden_states), dim=1
        )
        
        # Calculate entropy
        entropy = -torch.sum(
            attention_proxy * torch.log(attention_proxy + 1e-8), 
            dim=1
        ).mean(dim=-1)
        
        # Normalize entropy to [0, 1]
        normalized_entropy = torch.sigmoid(entropy)
        
        return normalized_entropy
```

---

## ⚙️ Implementation Details

### Adaptive K Calculator

```python
class AdaptiveKCalculator(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Learnable weights for different factors
        self.length_weight = nn.Parameter(torch.tensor(0.4))
        self.complexity_weight = nn.Parameter(torch.tensor(0.3))
        self.entropy_weight = nn.Parameter(torch.tensor(0.3))
        
        # Base sparsity level
        self.base_sparsity = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, seq_len, length_factor, complexity_factor, entropy_factor):
        # Weighted combination of factors
        adaptive_factor = (
            self.length_weight * length_factor +
            self.complexity_weight * complexity_factor +
            self.entropy_weight * entropy_factor
        )
        
        # Calculate k as fraction of sequence length
        k_fraction = self.base_sparsity * adaptive_factor
        
        # Ensure k is within reasonable bounds [0.1, 0.9]
        k_fraction = torch.clamp(k_fraction, 0.1, 0.9)
        
        # Convert to integer k
        k = torch.round(seq_len * k_fraction).long()
        
        # Ensure k is at least 1
        k = torch.clamp(k, 1, seq_len - 1)
        
        return k
```

### Integration with DeepSeek Sparse Attention

```python
class AdaptiveDeepSeekSparseAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        
        # Standard DeepSeek sparse attention components
        self.indexer = LightningIndexer(config)
        self.selector = TopKTokenSelector(config)
        self.attention = DeepseekV3Attention(config, layer_idx)
        
        # Dynamic sparsity controller
        self.sparsity_controller = DynamicSparsityController(
            config.d_model, config.max_position_embeddings
        )
        
    def forward(self, hidden_states, attention_mask=None):
        seq_len = hidden_states.size(1)
        
        # Calculate adaptive k
        adaptive_k = self.sparsity_controller(hidden_states, seq_len)
        
        # Compute indexer scores
        index_scores = self.indexer(hidden_states)
        
        # Select top-k tokens with adaptive k
        top_k_mask, selected_indices = self.selector(
            index_scores, k=adaptive_k
        )
        
        # Apply sparse attention
        sparse_output = self.attention(
            hidden_states, 
            attention_mask=create_sparse_mask(top_k_mask)
        )
        
        return sparse_output, {
            'adaptive_k': adaptive_k,
            'selected_indices': selected_indices,
            'sparsity_ratio': 1.0 - (adaptive_k.float() / seq_len)
        }
```

---

## 📊 Evaluation Metrics

### Primary Metrics (Speed & Quality)
1. **Training Speed**: Tokens/second, steps/second, wall-clock time
2. **Convergence Speed**: Steps to reach target loss
3. **Final Performance**: Validation loss, perplexity, accuracy
4. **Memory Efficiency**: Peak memory usage, memory per token

### Secondary Metrics (Adaptive Behavior)
1. **Sparsity Distribution**: k values across different sequence characteristics
2. **Adaptation Quality**: Correlation between predicted k and optimal k
3. **Stability**: Variance in k predictions across training
4. **Computational Overhead**: Cost of adaptive k calculation

### Statistical Analysis
1. **Multiple Runs**: 3 runs per configuration with different seeds
2. **Confidence Intervals**: 95% CI for all metrics
3. **Statistical Tests**: t-tests for significance testing
4. **Effect Size**: Cohen's d for practical significance

---

## 🎯 Expected Outcomes

### Speed Improvements
- **Target**: 10-30% faster training than fixed sparsity
- **Mechanism**: Adaptive k reduces computation on simple sequences
- **Measurement**: Wall-clock time per epoch

### Quality Improvements  
- **Target**: 5-15% better final performance than fixed sparsity
- **Mechanism**: Adaptive k preserves important tokens on complex sequences
- **Measurement**: Validation loss and perplexity

### Adaptation Insights
- **Length Adaptation**: Optimal k patterns for different sequence lengths
- **Complexity Adaptation**: How content complexity affects optimal sparsity
- **Entropy Adaptation**: Relationship between attention entropy and sparsity needs

---

## 🔬 Scientific Rigor

### Controls
- **Fixed Random Seeds**: Ensures reproducible results
- **Multiple Baselines**: Fixed sparsity levels (25%, 50%, 75%) + dense
- **Consistent Training**: Same hyperparameters, data, and training protocol
- **Statistical Significance**: Multiple runs with proper statistical testing

### Fairness
- **Parameter Budget**: All models have similar total parameter count
- **Training Budget**: Same number of training steps for all models
- **Evaluation Protocol**: Identical evaluation on same held-out data
- **Hardware Consistency**: Same GPU and batch size for all experiments

### Correctness
- **Implementation Verification**: Unit tests for all adaptive components
- **Gradient Flow**: Verify gradients flow through adaptive components
- **Numerical Stability**: Check for NaN/inf values during training
- **Memory Bounds**: Ensure adaptive k stays within reasonable bounds

### Measurement Validity
- **Multiple Metrics**: Speed, quality, and adaptation quality
- **Cross-Validation**: Results consistent across different sequence lengths
- **Ablation Studies**: Test individual adaptive components
- **Error Analysis**: Detailed analysis of failure cases

---

## 🚀 Implementation Plan

1. **Week 1**: Implement adaptive sparsity controller and basic components
2. **Week 2**: Integrate with DeepSeek sparse attention and test basic functionality  
3. **Week 3**: Run full experiments across all sequence lengths
4. **Week 4**: Analyze results, statistical testing, and documentation

---

## 📚 Related Work

- **DeepSeek-V3**: Multi-Head Latent Attention with fixed sparsity
- **Adaptive Attention**: Dynamic attention patterns in vision transformers
- **Efficient Transformers**: Various sparsity patterns for efficiency
- **Content-Aware Models**: Models that adapt based on input characteristics

---

## 🎓 Expected Contributions

1. **Novel Architecture**: First adaptive sparsity controller for transformer attention
2. **Empirical Insights**: Understanding of optimal sparsity patterns across sequence characteristics
3. **Practical Benefits**: Improved training speed and model quality
4. **Theoretical Understanding**: Relationship between sequence properties and attention needs

---

*Experiment designed: December 2024*  
*Focus: Dynamic sparsity patterns for optimal pretraining efficiency*
