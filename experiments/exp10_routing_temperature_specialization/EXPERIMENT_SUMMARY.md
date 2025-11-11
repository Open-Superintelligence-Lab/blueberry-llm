# Experiment 10: Routing Temperature & Expert Specialization Analysis

## Executive Summary

This experiment systematically explores how **routing temperature** affects Mixture-of-Experts (MoE) model training. Temperature controls the sharpness of the routing distribution, creating a fundamental trade-off between exploration (uniform routing) and exploitation (sharp, confident routing).

### Key Innovation

Unlike previous MoE work that uses fixed temperature (typically 1.0), we:
1. **Systematically ablate temperatures** from 0.5 (sharp) to 10.0 (uniform)
2. **Explore temperature scheduling** (high→low over training)
3. **Comprehensively track routing dynamics** (entropy, utilization, specialization)
4. **Generate rich visualizations** to understand expert behavior

## Motivation

**Why does temperature matter?**

In MoE models, the routing network uses softmax to assign tokens to experts:

```
router_probs = softmax(logits / temperature)
```

- **Low temperature (< 1.0)**: Sharp routing
  - ✅ Strong expert specialization
  - ✅ Fast convergence
  - ❌ Risk of load imbalance
  - ❌ Risk of premature specialization

- **High temperature (> 1.0)**: Soft routing  
  - ✅ Better load balancing
  - ✅ More exploration
  - ❌ Slower specialization
  - ❌ Potentially worse final performance

- **Temperature scheduling**: Best of both worlds?
  - Start high: Exploration phase, find good expert assignments
  - End low: Exploitation phase, refine specializations

## Experiment Design

### Model Architecture

```
MoE Transformer
├── Layers: 6
├── d_model: 384
├── n_heads: 8  
├── d_ff: 1536
├── Experts: 8
├── Top-k: 2
└── Total params: ~79M (28.4% active)
```

### Experiment Matrix

#### 1. Temperature Ablation (8 experiments × 500 steps)

| Experiment | Temperature | Description | Expected Behavior |
|------------|-------------|-------------|-------------------|
| temp_0.5   | 0.5 | Very sharp | Strong specialization, risk imbalance |
| temp_0.7   | 0.7 | Sharp | Moderate specialization |
| **temp_1.0** | **1.0** | **Baseline** | **Standard softmax** |
| temp_1.5   | 1.5 | Slightly soft | More balanced |
| temp_2.0   | 2.0 | Soft | Good exploration |
| temp_3.0   | 3.0 | Very soft | High exploration |
| temp_5.0   | 5.0 | Nearly uniform | Maximum exploration |
| temp_10.0  | 10.0 | Uniform | Minimal specialization |

#### 2. Temperature Scheduling (4 experiments × 500 steps)

| Experiment | Schedule | Formula | Rationale |
|------------|----------|---------|-----------|
| schedule_linear | Linear | `5.0 + (1.0 - 5.0) * progress` | Simple decay |
| schedule_cosine | Cosine | `1.0 + (5.0 - 1.0) * 0.5 * (1 + cos(π*progress))` | Smooth decay |
| schedule_exp | Exponential | `5.0 * (1.0/5.0)^progress` | Fast early decay |
| schedule_step | Step | `5.0 → 2.0 → 1.0` | Discrete phases |

#### 3. Extended Training (1 experiment × 1000 steps)

| Experiment | Temperature | Description |
|------------|-------------|-------------|
| temp_best_long | TBD (best from ablation) | Longer training with optimal temp |

**Total: 13 experiments**

### Training Configuration

Based on optimal settings from Experiment 9 (Muon vs Adam):

```python
Optimizer:           Muon (hybrid)
  muon_lr:          0.07
  adamw_lr:         0.007
  momentum:         0.9
  weight_decay:     0.2
  
Training:
  steps:            500 (1000 for extended)
  batch_size:       24
  grad_accumulation: 4
  effective_batch:  96
  
LR Schedule:        Cosine decay
  warmup_ratio:     0.05 (25 steps)
  min_lr_ratio:     0.1
  
Regularization:
  dropout:          0.1
  grad_clip:        1.0
  load_bal_weight:  0.01
  
Data:               HuggingFaceTB/smollm-corpus (cosmopedia-v2)
  train_docs:       1,800
  val_docs:         200
  seq_length:       512 tokens
```

## Metrics & Analysis

### Primary Metrics

**Performance Metrics:**
- Validation loss (primary objective)
- Validation accuracy
- Validation perplexity
- Training time (wall-clock)

**Routing Metrics:**
- **Routing entropy**: Measures routing diversity
  - High entropy = uniform routing
  - Low entropy = sharp routing
  - Formula: `H = -Σ p_i log(p_i)`

- **Selection confidence**: How strongly top expert is preferred
  - Range: [0.5, 1.0] for top-2 routing
  - Higher = more confident routing

- **Expert utilization**: Fraction of tokens per expert
  - Ideal: 1/8 = 0.125 (uniform)
  - Gini coefficient: measures inequality

- **Load balancing loss**: Auxiliary loss to encourage balance
  - Lower = better load distribution

### Visualization Suite

Our comprehensive analysis generates:

#### 1. Temperature Comparison Plots
- Loss curves for all temperatures
- Performance vs temperature (log scale)
- Accuracy vs temperature
- Routing entropy vs temperature
- Summary statistics table

#### 2. Routing Dynamics Analysis
- Entropy evolution over training
- Selection confidence evolution
- Load balancing loss trends
- Temperature vs final routing metrics

#### 3. Expert Utilization Patterns
- Per-expert utilization bars for each temperature
- Heatmap: temperatures × experts
- Gini coefficient analysis
- Utilization variance analysis

#### 4. Schedule Comparison
- Loss/accuracy curves for all schedules
- Temperature evolution visualization
- Final performance comparison

#### 5. Specialization Analysis
- Gini coefficient vs temperature
- Utilization variance vs temperature
- Expert activation heatmaps
- Entropy change rate analysis

## Expected Results & Hypotheses

### Hypothesis 1: Optimal Temperature ≈ 1.5-2.0

**Reasoning:**
- temp = 1.0 is arbitrary (just standard softmax)
- Slightly higher temp should improve exploration
- Too high temp loses specialization benefits

**Expected curve:**
```
Loss
 ^
 |     *
 |    * *
 |   *   *
 |  *     *
 | *       *
 +-----------> Temperature
 0.5  1.0  2.0  5.0  10.0
```

### Hypothesis 2: Temperature Scheduling Helps

**Reasoning:**
- Early training: Need exploration to find good expert assignments
- Late training: Need exploitation to refine specializations
- Cosine schedule likely best (smooth transition)

**Expected ranking:**
1. Cosine schedule (smooth decay)
2. Exponential schedule (fast early exploration)
3. Step schedule (abrupt transitions)
4. Linear schedule (too slow early)

### Hypothesis 3: Low Temp → Load Imbalance

**Reasoning:**
- Sharp routing leads to winner-take-all dynamics
- Some experts become dominant, others unused
- Higher Gini coefficient, higher load balancing loss

**Expected:**
- temp=0.5: High Gini (> 0.3), poor load balancing
- temp=1.0: Moderate Gini (~0.2)
- temp=2.0: Low Gini (< 0.15), good balance

### Hypothesis 4: Entropy Decreases Over Training

**Reasoning:**
- Early training: High uncertainty, higher entropy
- Late training: Experts specialize, lower entropy
- Effect more pronounced at lower temperatures

## Implementation Details

### Key Components

#### 1. `TemperatureRouter` (temperature_router.py)
```python
class TemperatureRouter(nn.Module):
    def forward(self, x):
        logits = self.gate(x)
        scaled_logits = logits / self.current_temperature
        probs = softmax(scaled_logits)
        # ... track statistics
```

**Features:**
- Dynamic temperature setting
- Comprehensive routing statistics
- Entropy & confidence tracking
- Expert utilization monitoring

#### 2. `TemperatureMoE` (temperature_moe.py)
```python
class TemperatureMoE(nn.Module):
    def set_temperature(self, temp):
        self.router.set_temperature(temp)
    
    def forward(self, x, return_routing_stats=True):
        # Route tokens to experts
        # Track activation patterns
        # Return stats if requested
```

**Features:**
- Temperature-aware routing
- Expert activation tracking
- Detailed routing statistics
- Load balancing loss computation

#### 3. `train_with_temperature_tracking` (tracking_trainer.py)
```python
def train_with_temperature_tracking(...):
    for step in range(max_steps):
        # Update temperature (scheduled or constant)
        current_temp = temp_config.get_temperature_at_step(step)
        model.set_temperature(current_temp)
        
        # Training step
        # Collect routing stats at eval points
        # Save comprehensive history
```

**Features:**
- Temperature scheduling support
- Comprehensive metric tracking
- Routing statistics collection
- Rich training history

### Data Flow

```
Input Tokens
    ↓
[Token Embeddings]
    ↓
[Attention Layer]
    ↓
[RMS Norm]
    ↓
┌─────────────────────────┐
│  TemperatureMoE Layer   │
│                         │
│  [TemperatureRouter]    │
│    - Apply temperature  │
│    - Compute probs      │
│    - Select top-k       │
│    - Track stats        │
│         ↓               │
│  [Expert Processing]    │
│    - Route tokens       │
│    - Apply experts      │
│    - Weighted combine   │
└─────────────────────────┘
    ↓
[RMS Norm]
    ↓
[LM Head]
    ↓
Logits + Aux Loss + Routing Stats
```

## Usage Examples

### Basic Usage

```bash
# List all experiments
python run_experiment.py --list

# Run single temperature
python run_experiment.py --experiment temp_1.0

# Run temperature ablation (8 experiments)
python run_experiment.py --ablation

# Run temperature schedules (4 experiments)
python run_experiment.py --schedules

# Run all 13 experiments
python run_experiment.py --all

# Custom temperature
python run_experiment.py --temperature 1.5
```

### Quick Demo

```bash
# Run 3 representative temperatures (temp_0.7, temp_1.0, temp_2.0)
bash quick_demo.sh
```

This runs 3 experiments (500 steps each, ~2-3 min per experiment) and generates all visualizations.

### Analysis & Visualization

```bash
# Generate comprehensive plots
python plot_results.py \
    --results-dir ./results \
    --output-dir ./analysis

# Analyze expert specialization
python analyze_specialization.py \
    --results-dir ./results \
    --output-dir ./analysis
```

### Output Structure

```
exp10_routing_temperature_specialization/
├── results/
│   ├── temp_0.5/
│   │   ├── metrics.json           # Complete training history
│   │   ├── model.pt               # Model checkpoint
│   │   └── logs/                  # Training logs
│   ├── temp_1.0/
│   │   └── ...
│   └── ...
│
└── analysis/
    ├── temperature_ablation_comprehensive.png
    ├── routing_dynamics.png
    ├── expert_utilization.png
    ├── expert_utilization_analysis.png
    ├── entropy_analysis.png
    ├── schedule_comparison.png
    ├── summary_report.json
    └── specialization_report.json
```

## Knowledge Generated

This experiment will yield deep insights into:

### 1. **Optimal Temperature Discovery**
- Empirically determine best temperature for MoE training
- Understand temperature-performance relationship
- Quantify sensitivity to temperature choice

### 2. **Routing Dynamics Understanding**
- How routing evolves during training
- When/how experts specialize
- Impact of temperature on specialization patterns

### 3. **Load Balancing Insights**
- Trade-off between specialization and balance
- Effectiveness of load balancing loss at different temperatures
- Alternative approaches to temperature for better balance

### 4. **Schedule Design Principles**
- Does scheduling help? By how much?
- What schedule shape is optimal?
- When to transition from exploration to exploitation?

### 5. **Practical Guidelines**
- Actionable recommendations for MoE practitioners
- Temperature tuning as hyperparameter
- Integration with other training techniques

## Extensions & Future Work

### Immediate Extensions

1. **Longer Training**: Run best temperature for 5k-10k steps
2. **Larger Models**: Scale to 1B+ parameters
3. **Different Architectures**: Test with different expert/attention designs
4. **Per-Layer Temperatures**: Different temps for different layers

### Research Directions

1. **Adaptive Temperature**: Learn temperature during training
2. **Token-Dependent Temperature**: Different temps for different tokens
3. **Expert-Specific Temperature**: Per-expert routing sharpness
4. **Annealing Strategies**: More sophisticated scheduling
5. **Uncertainty-Based Routing**: Use temperature to model uncertainty

### Integration Opportunities

1. **Combine with Exp9**: Optimal optimizer + optimal temperature
2. **Architecture Search**: Find best temp for different architectures  
3. **Dataset Effects**: How does optimal temp vary by dataset?
4. **Multi-Objective**: Balance loss, speed, and expert utilization

## Technical Notes

### Reproducibility

- All experiments use seed=42
- Data split before tokenization (no leakage)
- Deterministic operations where possible
- Complete configuration saved with results

### Computational Requirements

- **Per experiment**: ~2-3 minutes on GPU (V100/A100)
- **Full ablation**: ~20-25 minutes (8 experiments)
- **Complete suite**: ~40-50 minutes (13 experiments)
- **Memory**: ~8-10 GB GPU RAM

### Implementation Choices

**Why these temperatures?**
- 0.5, 0.7: Sharp routing regime
- 1.0: Standard baseline
- 1.5, 2.0, 3.0: Exploration regime  
- 5.0, 10.0: Extreme exploration

**Why these schedules?**
- Linear: Simple baseline
- Cosine: Smooth, widely used
- Exponential: Fast early exploration
- Step: Test discrete phase transitions

**Why 500 steps?**
- Fast iteration for ablation
- Sufficient to see convergence trends
- Extended run (1000 steps) validates best setting

## Conclusion

Experiment 10 represents a **comprehensive, systematic exploration** of routing temperature in MoE models—a fundamental but under-studied hyperparameter. Through careful ablation, rich tracking, and extensive visualization, we will:

1. ✅ Discover optimal temperature for MoE training
2. ✅ Understand exploration-exploitation trade-offs
3. ✅ Quantify impact on expert specialization
4. ✅ Develop practical temperature scheduling strategies
5. ✅ Generate actionable insights for MoE practitioners
6. ✅ Create comprehensive visualization toolkit
7. ✅ Build foundation for future routing research

The experiment is **ready to run** and will generate **significant new knowledge** about MoE training dynamics.

---

**Created**: November 11, 2025  
**Branch**: `exp10-routing-temperature-analysis`  
**Status**: ✅ Ready to execute  
**Estimated Time**: 40-50 minutes for complete suite

