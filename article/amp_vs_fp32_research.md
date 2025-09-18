# 🔬 Research: When is Mixed Precision (AMP) Faster than FP32 on T4 GPU?

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Automatic Mixed Precision (AMP)](#understanding-automatic-mixed-precision-amp)
3. [Blueberry-LLM T4 Configuration](#blueberry-llm-t4-configuration)
4. [What Happens When AMP is Enabled](#what-happens-when-amp-is-enabled)
5. [Research Methodology](#research-methodology)
6. [Experimental Setup](#experimental-setup)
7. [Results and Analysis](#results-and-analysis)
8. [Conclusions and Recommendations](#conclusions-and-recommendations)
9. [References](#references)

---

## Introduction

This research investigates the performance characteristics of Automatic Mixed Precision (AMP) versus full precision (FP32) training on T4 GPUs, specifically in the context of the **blueberry-llm** project's auto-configuration system. While AMP is theoretically designed to accelerate training through reduced memory usage and faster computation, initial observations suggest that performance gains are not universal and may depend on specific model characteristics and training configurations.

**Research Question:** Under what conditions does AMP actually improve performance on a T4 GPU in the blueberry-llm framework?

**Key Focus:** Understanding how the blueberry-llm auto-configuration system adapts to T4 GPUs and when AMP provides actual speedups versus overhead.

---

## Understanding Automatic Mixed Precision (AMP)

### What is Mixed Precision Training?

Automatic Mixed Precision (AMP) is a training technique that combines different numerical precisions within a single training process to optimize both computational speed and memory efficiency. The core idea is to use **half-precision (FP16)** for most operations while maintaining **full-precision (FP32)** for operations that require higher numerical stability.

![AMP Overview](./images/amp_overview.png)
*Figure 1: Conceptual overview of Mixed Precision training*

### Key Components of AMP

#### 1. **Forward Pass in FP16**
- Most computations (matrix multiplications, convolutions) are performed in FP16
- Significantly faster than FP32 on modern GPUs
- Reduced memory bandwidth requirements

#### 2. **Loss Scaling**
- FP16 has a limited dynamic range (6 orders of magnitude vs 38 for FP32)
- Gradient values can underflow (become zero) during backpropagation
- Loss scaling multiplies the loss by a large factor before backpropagation
- Gradients are unscaled before optimizer updates

![Loss Scaling Process](./images/loss_scaling_process.png)
*Figure 2: Loss scaling mechanism in AMP*

#### 3. **Master Weights in FP32**
- Model parameters are stored in FP32 for numerical stability
- FP16 copies are used for forward/backward passes
- Updates are applied to FP32 master weights

#### 4. **Automatic Casting**
- Framework automatically determines which operations need FP32
- Operations like loss computation, batch normalization, and certain activations remain in FP32

### Mathematical Foundation

The core mathematical operations in AMP can be represented as:

#### **1. Forward Pass**
```
y = f(x; θ_fp16)
```
**Explanation:** The forward pass computes model outputs using FP16 parameters. Here, `f` represents the neural network function (attention, feed-forward layers, etc.), `x` is the input tensor, and `θ_fp16` are the model parameters stored in FP16 precision. This operation benefits from FP16's faster computation and reduced memory bandwidth.

θ is sign for parameters, so this just means forward pass with input x where parameters are θ_fp16.

#### **2. Loss Computation**
```
L = CrossEntropy(y, y_true)
```
**Explanation:** The loss is computed using the FP16 forward pass outputs. Cross-entropy loss measures the difference between predicted probabilities (`y`) and true labels (`y_true`). This computation remains in FP16 for efficiency, but the loss value may be very small and prone to underflow.

#### **3. Loss Scaling**
```
L_scaled = L(y, y_true) × scale_factor
```
**Explanation:** To prevent gradient underflow, the loss is multiplied by a large scaling factor (typically 2^16 = 65536). This shifts the loss into a range where FP16 can represent it without losing precision. The scaling factor is dynamically adjusted based on gradient overflow detection.

**FP16 Range Details:** FP16 has a limited dynamic range (approximately 6 orders of magnitude vs 38 for FP32), so very small loss values can underflow to zero. The scaling factor shifts these values into FP16's representable range. To understand how these range conversions work in practice, examine the `GradScaler` implementation in PyTorch's `torch.cuda.amp` module, which handles the automatic detection of overflow/underflow and dynamic scaling adjustments.

As this is a separate topic in itself, you can copy this into AI chatbot and self study it.

#### **4. Backward Pass**
```
∇θ_fp16 = ∇L_scaled / scale_factor
```
**Explanation:** Gradients are computed using the scaled loss, then divided by the same scaling factor to restore the correct magnitude. This ensures gradients maintain their proper scale while being computed in FP16 precision. The division is crucial for numerical stability.

#### **5. Gradient Clipping (Optional)**
```
∇θ_fp16 = clip(∇θ_fp16, max_norm)
```
**Explanation:** Gradients are clipped to prevent exploding gradients, which is especially important in mixed precision training. In FP16, the smaller representable range and lower precision can amplify issues like gradient overflow or underflow, making numerical instability more likely compared to FP32. Clipping helps mitigate these risks by ensuring gradients stay within a safe, predefined range, thus maintaining training stability.

#### **6. Weight Update**
```
θ_fp32 = θ_fp32 - α × ∇θ_fp16
```
**Explanation:** The FP32 master weights are updated using FP16 gradients. The learning rate `α` controls the step size. This step maintains numerical precision by keeping the master weights in FP32 while using FP16 gradients for efficiency.

#### **7. Parameter Synchronization**
```
θ_fp16 = θ_fp32  (copy to FP16)
```
**Explanation:** After updating FP32 master weights, they are copied back to FP16 for the next forward pass. This ensures the FP16 parameters used in computation are synchronized with the FP32 master weights.

#### **8. Dynamic Scaling Adjustment**
```
if overflow_detected:
    scale_factor = scale_factor / 2
elif no_overflow_for_N_steps:
    scale_factor = scale_factor × 2
```
**Explanation:** The scaling factor is dynamically adjusted based on gradient overflow detection. If overflow occurs, the scale is reduced to prevent future overflows. If no overflow occurs for several steps, the scale is increased to maximize precision.

**Key Variables:**
- `θ_fp16`: FP16 model parameters (used in forward/backward passes)
- `θ_fp32`: FP32 master weights (used for parameter updates)
- `scale_factor`: Dynamic loss scaling factor (typically 2^16, adjusted automatically)
- `α`: Learning rate (controls update step size)
- `max_norm`: Gradient clipping threshold (prevents exploding gradients)
- `N`: Steps without overflow before increasing scale factor

---

## Theoretical Performance Benefits

### Memory Efficiency

FP16 uses **half the memory** of FP32:
- **FP32**: 32 bits (4 bytes) per parameter
- **FP16**: 16 bits (2 bytes) per parameter

This enables:
- **Larger batch sizes** for the same memory budget
- **Training larger models** on limited GPU memory
- **Better memory bandwidth utilization**

![Memory Comparison](./images/memory_comparison.png)
*Figure 3: Memory usage comparison between FP32 and FP16*

### Computational Speed

Modern GPUs (including T4) have specialized FP16 units:
- **Tensor Cores** (T4): Optimized for mixed-precision operations
- **Higher throughput** for FP16 operations
- **Reduced power consumption**

### Theoretical Speedup Factors

1. **Memory Bandwidth**: ~2x improvement
2. **Compute Throughput**: 1.5-2x improvement (depending on operation)
3. **Combined Effect**: Up to 2-4x theoretical speedup

---

## Blueberry-LLM T4 Configuration

### Auto-Detection System

The blueberry-llm project includes an intelligent auto-configuration system that specifically detects and optimizes for T4 GPUs. When you run `python core/auto_config.py` or `python train_auto.py`, the system:

1. **Detects T4 GPU**: Identifies Tesla T4 or T4 architecture
2. **Applies T4-Optimized Settings**: Automatically configures model parameters
3. **Enables AMP by Default**: Sets `use_amp=True` for T4 optimization

![T4 Auto-Configuration](./images/t4_auto_config.png)
*Figure 4: T4 detection and auto-configuration flow*

### T4-Specific Configuration

When a T4 GPU is detected, the system applies these optimized settings:

```python
# T4 Optimized Configuration (from auto_config.py)
AutoConfig(
    d_model=384,           # Moderate size for T4 memory
    n_layers=6,            # Balanced depth
    n_heads=8,             # Optimized attention heads
    d_ff=1536,             # Feed-forward dimension
    num_experts=8,         # MoE experts
    batch_size=12,         # Memory-efficient batch size
    gradient_accumulation_steps=3,
    max_steps=2000,
    learning_rate=0.01,
    max_seq_len=1024,
    use_amp=True,          # AMP enabled by default
    use_megatron=False     # Native PyTorch backend
)
```

### System Architecture Detection

The system classifies T4 as **"ampere"** architecture with:
- **Compute Capability**: (8, 0) - Ampere generation
- **Tensor Cores**: ✅ Supported
- **BF16 Support**: ✅ Supported  
- **FP8 Support**: ❌ Not supported (Blackwell+ only)
- **Memory**: ~16GB VRAM

![T4 Architecture Features](./images/t4_architecture_features.png)
*Figure 5: T4 GPU capabilities and supported features*

---

## What Happens When AMP is Enabled

### Automatic Configuration Changes

When `use_amp=True` is set (default for T4), several key changes occur:

#### 1. **Gradient Scaler Initialization**
```python
# From training/trainer.py and legacy/llm.py
scaler = GradScaler('cuda') if config.use_amp else None
```

#### 2. **Forward Pass with Autocast**
```python
# All forward passes wrapped in autocast
with autocast(enabled=config.use_amp):
    logits = model(x, return_aux_loss=False)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
```

#### 3. **Backward Pass with Scaling**
```python
# Gradient scaling for numerical stability
if config.use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

### Memory Usage Impact

**With AMP Enabled:**
- **Model Parameters**: Stored in FP32 (master weights)
- **Activations**: Computed in FP16 during forward pass
- **Gradients**: Computed in FP16, scaled for stability
- **Memory Savings**: ~50% reduction in activation memory

**Memory Breakdown for T4 Configuration:**
```
Model Size: 384d × 6L × 8H = ~50M parameters
- FP32 Master Weights: ~200MB
- FP16 Activations: ~100MB (vs 200MB FP32)
- FP16 Gradients: ~100MB (vs 200MB FP32)
- Total Memory Savings: ~200MB per batch
```

![Memory Usage Comparison](./images/memory_usage_comparison.png)
*Figure 6: Memory usage comparison on T4 with AMP vs FP32*

### Performance Characteristics

#### **Expected Benefits on T4:**
1. **Memory Bandwidth**: 2x improvement (FP16 vs FP32)
2. **Tensor Core Utilization**: Better utilization of T4's tensor cores
3. **Batch Size Scaling**: Can potentially increase batch size
4. **Training Speed**: 1.5-2x theoretical speedup

#### **Potential Overhead:**
1. **Type Conversion**: FP32 ↔ FP16 conversions
2. **Loss Scaling**: Additional scaling/unscaling operations
3. **Small Model Penalty**: Overhead may exceed benefits for very small models
4. **Memory Fragmentation**: Mixed precision can cause memory fragmentation

### Training Pipeline Changes

When AMP is enabled, the training pipeline includes:

1. **Automatic Type Casting**: PyTorch automatically casts operations to FP16
2. **Loss Scaling**: Dynamic loss scaling to prevent gradient underflow
3. **Master Weight Updates**: FP32 master weights updated with FP16 gradients
4. **Numerical Stability**: Automatic handling of overflow/underflow

![AMP Training Pipeline](./images/amp_training_pipeline.png)
*Figure 7: Complete AMP training pipeline on T4*

---

## Research Methodology

### Experimental Design

This research will systematically investigate AMP performance using the blueberry-llm framework across different model configurations:

#### **Independent Variables:**
1. **Model Size** (using blueberry-llm's auto-config system)
   - `d_model`: [128, 256, 384, 512] (T4-optimized range)
   - `n_layers`: [2, 4, 6, 8, 12] (T4 memory constraints)
   - `num_experts`: [4, 8, 16] (MoE scaling)

2. **Training Configuration**
   - `batch_size`: [8, 12, 16, 24] (T4 memory limits)
   - `sequence_length`: [256, 512, 1024] (T4 optimal range)
   - `gradient_accumulation_steps`: [1, 2, 3, 4]

3. **Precision Mode**
   - FP32 (baseline): `use_amp=False`
   - AMP (experimental): `use_amp=True` (default T4 setting)

#### **Dependent Variables:**
- Training time per epoch
- Memory usage (peak GPU memory)
- Throughput (tokens/second)
- Convergence behavior
- Final model accuracy

### Measurement Methodology

#### **Performance Metrics:**
1. **Wall-clock time** for complete training epochs
2. **GPU memory utilization** (peak and average)
3. **GPU utilization** percentage
4. **Throughput** in tokens processed per second

#### **Statistical Analysis:**
- Multiple runs (minimum 3) per configuration
- Confidence intervals for timing measurements
- Statistical significance testing

---

## Experimental Setup

### Hardware Configuration
- **GPU**: NVIDIA T4 (16GB VRAM)
- **Environment**: Google Colab Pro / Local T4 setup
- **CUDA**: Version 11.8+
- **PyTorch**: Version 2.0+

### Blueberry-LLM Testing Framework

The experimental framework leverages blueberry-llm's built-in capabilities:

#### **1. Auto-Configuration Testing**
```bash
# Test T4 auto-detection
python core/auto_config.py

# Run with AMP enabled (default)
python train_auto.py

# Run with AMP disabled
python train_auto.py --no-amp
```

#### **2. Custom Configuration Testing**
```python
# Modify auto_config.py for specific experiments
def _t4_optimized_config(self, num_gpus: int, gpu_memory_gb: float) -> AutoConfig:
    return AutoConfig(
        d_model=384,  # Test different sizes
        n_layers=6,   # Test different depths
        batch_size=12,  # Test different batch sizes
        use_amp=True,   # Enable/disable AMP
        # ... other parameters
    )
```

#### **3. Performance Monitoring**
The framework includes built-in GPU monitoring:
- **Memory Usage**: Peak and average GPU memory
- **Training Speed**: Tokens per second
- **Loss Scaling**: AMP gradient scaling statistics
- **Convergence**: Training/validation loss curves

#### **4. Benchmarking Scripts**
```python
# Example benchmarking approach
def benchmark_amp_vs_fp32():
    configs = [
        {"use_amp": False, "d_model": 384, "batch_size": 12},  # FP32 baseline
        {"use_amp": True, "d_model": 384, "batch_size": 12},   # AMP
        {"use_amp": True, "d_model": 384, "batch_size": 16},   # AMP + larger batch
    ]
    
    for config in configs:
        # Run training and measure performance
        metrics = run_training_experiment(config)
        log_results(config, metrics)
```

### Software Stack
```python
# Key dependencies (from requirements.txt)
torch>=2.0.0
torch.cuda.amp  # Automatic Mixed Precision
nvidia-ml-py3   # GPU monitoring
psutil          # System monitoring
matplotlib      # Visualization
tqdm            # Progress bars
```

### Testing Methodology

1. **Baseline Measurement**: Run FP32 training with T4-optimized config
2. **AMP Comparison**: Run identical config with AMP enabled
3. **Scaling Tests**: Test different model sizes and batch sizes
4. **Memory Analysis**: Monitor peak memory usage and efficiency
5. **Convergence Analysis**: Compare training dynamics and final accuracy

---

## Results and Analysis

*This section will be populated with experimental results as the research progresses.*

### Expected Analysis Areas

1. **Performance Scaling with Model Size**
2. **Memory Efficiency Gains**
3. **Batch Size Optimization**
4. **Sequence Length Impact**
5. **Convergence Behavior Comparison**

---

## Conclusions and Recommendations

*Conclusions will be drawn based on experimental findings.*

### Anticipated Insights

1. **Optimal Model Sizes** for AMP benefits
2. **Memory Thresholds** where AMP becomes advantageous
3. **Training Configuration** recommendations
4. **When to Avoid AMP** scenarios

---

## References

1. Micikevicius, P., et al. "Mixed precision training." ICLR 2018.
2. NVIDIA. "Training with Mixed Precision." NVIDIA Developer Documentation.
3. PyTorch. "Automatic Mixed Precision." PyTorch Documentation.

---

## Appendix

### Code Repository
- **Branch**: `research/t4-amp-vs-fp32`
- **Experiment Scripts**: `article/experiments/`
- **Results Data**: `article/results/`
- **Visualizations**: `article/images/`

### Contributing
This research is part of the [blueberry-llm](https://github.com/Open-Superintelligence-Lab/blueberry-llm) project. Contributions and feedback are welcome through GitHub issues and pull requests.

---

*Last Updated: [Date will be updated as research progresses]*
*Author: Research conducted as part of Issue #19*
