# 🚀 Advanced GPU Profiler for Blueberry LLM

The Advanced GPU Profiler is a comprehensive performance monitoring system designed specifically for optimizing Mixture of Experts (MoE) model training and inference. It tracks memory allocation patterns, kernel execution times, data movement costs, and expert routing efficiency.

## 🎯 Features

### 📊 Memory Allocation Pattern Tracking
- **Real-time memory monitoring** - Track GPU memory usage, fragmentation, and allocation efficiency
- **Expert-specific memory tracking** - Monitor memory usage per expert
- **Memory leak detection** - Identify potential memory leaks and inefficiencies
- **Peak memory analysis** - Track maximum memory usage during training

### ⚡ Kernel Execution Time Monitoring
- **CUDA kernel profiling** - Monitor individual kernel execution times
- **Expert-specific performance** - Track performance metrics per expert
- **Kernel occupancy analysis** - Estimate kernel occupancy and efficiency
- **Memory bandwidth utilization** - Monitor memory bandwidth per kernel

### 🔄 Data Movement Cost Analysis
- **PCIe bandwidth monitoring** - Track host-to-device data transfers
- **GPU memory bandwidth** - Monitor device memory bandwidth utilization
- **Cache hit rate analysis** - Track cache efficiency
- **Data transfer optimization** - Identify data movement bottlenecks

### 🧠 Expert Routing Efficiency Monitoring
- **Load balancing analysis** - Monitor expert utilization distribution
- **Routing decision quality** - Track routing efficiency metrics
- **Imbalanced expert detection** - Identify underutilized or overloaded experts
- **Token distribution analysis** - Monitor token routing patterns

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA capability

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies
For enhanced profiling capabilities:
```bash
pip install nvidia-ml-py3  # For detailed GPU metrics
pip install matplotlib     # For visualization
pip install pandas        # For data analysis
```

## 🚀 Quick Start

### Basic Usage
```python
from advanced_gpu_profiler import AdvancedGPUProfiler, ProfilerContext

# Create profiler
profiler = AdvancedGPUProfiler(num_experts=8, enable_profiling=True)

# Use context manager for automatic start/stop
with ProfilerContext(profiler) as p:
    # Your training code here
    model = MoEMinimalLLM(config, profiler=p)
    # ... training loop ...
    
    # Print real-time dashboard
    p.print_dashboard()
```

### Integration with Training
```python
# Enable profiling in training
model, final_metrics = train_moe_model(
    config, train_loader, val_loader, 
    enable_profiling=True
)
```

### Standalone Profiling
```python
# Manual profiling control
profiler = AdvancedGPUProfiler(num_experts=8)
profiler.start_profiling()

# Your operations
profiler.profile_memory_allocation(size_bytes, expert_id=0, operation="attention")
profiler.profile_kernel_execution("attention_kernel", expert_id=0, operation_type="attention")
profiler.profile_expert_routing([0, 1], token_count=10)

# Print dashboard
profiler.print_dashboard()

# Stop and generate reports
profiler.stop_profiling()
```

## 📊 Real-time Dashboard

The profiler provides a comprehensive real-time dashboard:

```
🫐 Advanced GPU Profiler - Blueberry LLM
==========================================
📊 Memory Analysis:
  Peak Usage: 12.3GB / 24GB (51.2%)
  Fragmentation: 2.1% (Low)
  Expert Memory: [1.2GB, 1.1GB, 1.3GB, ...]
  
⚡ Kernel Performance:
  Attention Kernels: 2.3ms avg
  Expert Kernels: 1.8ms avg
  Memory Bandwidth: 78% utilized
  
🔄 Data Movement:
  PCIe Bandwidth: 45% utilized
  GPU Memory BW: 82% utilized
  Prefetch Hit Rate: 89%
  
🧠 Expert Routing:
  Expert Utilization: [85%, 92%, 78%, 88%, ...]
  Load Balance Score: 0.87 (Good)
  Routing Efficiency: 94%
```

## 📈 Generated Reports

The profiler automatically generates comprehensive reports:

### JSON Report
```json
{
  "timestamp": "2024-01-15_14:30:25",
  "profiling_duration": 120.5,
  "stats": {
    "memory": {
      "peak_memory_mb": 12345.6,
      "fragmentation_percent": 2.1,
      "expert_memory_distribution": [1200.5, 1100.2, 1300.8, ...]
    },
    "kernel": {
      "average_kernel_times": {
        "attention": 2.3,
        "expert_0": 1.8,
        "expert_1": 1.9,
        ...
      }
    },
    "expert_routing": {
      "expert_utilization": [85.2, 92.1, 78.5, 88.3, ...],
      "load_balance_score": 0.87,
      "routing_efficiency": 94.2
    }
  }
}
```

### CSV Reports
- `memory_stats_YYYYMMDD_HHMMSS.csv` - Memory usage over time
- `kernel_stats_YYYYMMDD_HHMMSS.csv` - Kernel execution times
- `expert_routing_YYYYMMDD_HHMMSS.csv` - Expert routing patterns

### Visualization Plots
- `memory_usage_YYYYMMDD_HHMMSS.png` - Memory usage trends
- `expert_utilization_YYYYMMDD_HHMMSS.png` - Expert utilization distribution

## 🔧 Configuration

### Profiler Configuration
```python
profiler = AdvancedGPUProfiler(
    num_experts=8,                    # Number of experts in MoE model
    enable_profiling=True,            # Enable/disable profiling
    output_dir="profiler_output",     # Output directory for reports
    max_history=1000                  # Maximum history entries to keep
)
```

### Integration Options
```python
# In MoE model initialization
model = MoEMinimalLLM(config, profiler=profiler)

# In training function
train_moe_model(config, train_loader, val_loader, enable_profiling=True)
```

## 🧪 Testing

Run the test suite to verify profiler functionality:

```bash
python test_profiler.py
```

The test suite includes:
- Basic profiler functionality
- Individual component testing
- Integration testing
- Performance validation

## 📊 Performance Impact

The profiler is designed for minimal overhead:
- **< 5% performance impact** during profiling
- **Asynchronous data collection** to minimize training disruption
- **Configurable profiling levels** for different use cases
- **Memory-efficient data storage** with automatic cleanup

## 🔍 Troubleshooting

### Common Issues

1. **ImportError: No module named 'advanced_gpu_profiler'**
   ```bash
   # Ensure you're in the project directory
   cd /path/to/blueberry-llm
   python -c "import advanced_gpu_profiler"
   ```

2. **CUDA not available**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Missing dependencies**
   ```bash
   # Install missing packages
   pip install nvidia-ml-py3 matplotlib pandas
   ```

### Debug Mode
```python
# Enable debug output
profiler = AdvancedGPUProfiler(num_experts=8, enable_profiling=True)
profiler.debug = True  # Enable detailed logging
```

## 🚀 Advanced Usage

### Custom Profiling Hooks
```python
# Add custom profiling to your code
@profile_function(profiler, "custom_operation")
def my_custom_function():
    # Your code here
    pass
```

### Performance Analysis
```python
# Get detailed performance metrics
stats = profiler.get_current_stats()
memory_efficiency = stats['memory']
kernel_performance = stats['kernel']
routing_efficiency = stats['expert_routing']
```

### Export Data for Analysis
```python
# Export profiling data for external analysis
profiler.generate_reports()
# Reports saved to profiler_output/ directory
```

## 🔮 Future Enhancements

- **Machine learning-based performance prediction**
- **Automatic optimization suggestions**
- **Cross-hardware performance comparison**
- **Integration with distributed training monitoring**
- **Real-time optimization recommendations**

## 📝 Contributing

We welcome contributions to improve the profiler! Areas for contribution:
- Additional profiling metrics
- Performance optimizations
- Visualization improvements
- Integration with other frameworks
- Documentation improvements

## 📄 License

This profiler is part of the Blueberry LLM project and follows the same license terms.

---

**Happy Profiling!** 🫐⚡📊

For questions or issues, please open a GitHub issue or check the main Blueberry LLM documentation.
