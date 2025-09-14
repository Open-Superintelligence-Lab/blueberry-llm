# 🫐 Blueberry LLM Project Structure

This document explains the organized folder structure of Blueberry LLM.

## 📁 Project Structure

```
blueberry-llm/
├── 📄 Core Files
│   ├── llm.py                    # Main MoE model implementation
│   ├── train_auto.py             # Auto-configured training script
│   ├── inference.py              # Text generation and inference
│   ├── auto_config.py            # Hardware auto-detection and configuration
│   ├── gpu_monitor.py            # Basic GPU monitoring
│   └── setup.sh                  # One-click setup script
│
├── 📊 profiler/                  # Advanced GPU Profiler
│   ├── __init__.py               # Profiler package exports
│   ├── advanced_gpu_profiler.py  # Core profiling engine
│   └── profiler_hooks.py         # Clean integration system
│
├── 🧪 tests/                     # Test Suite
│   ├── __init__.py               # Test package
│   └── test_profiler.py          # Profiler functionality tests
│
├── 📚 examples/                  # Example Scripts
│   ├── __init__.py               # Examples package
│   └── train_with_profiler.py    # Profiling integration example
│
├── 📖 docs/                      # Documentation
│   └── PROFILER_README.md        # Comprehensive profiler documentation
│
├── 🔧 Configuration
│   ├── requirements.txt          # Python dependencies
│   ├── PROJECT_STRUCTURE.md      # This file
│   └── README.md                 # Main project documentation
│
└── 📁 scripts/                   # Utility Scripts
    └── kubernetes/               # Kubernetes deployment scripts
```

## 🎯 Quick Start Guide

### 1. **First Time Setup**
```bash
git clone https://github.com/Open-Superintelligence-Lab/blueberry-llm
cd blueberry-llm
chmod +x setup.sh
./setup.sh
```

### 2. **Train Your First Model**
```bash
python train_auto.py
```

### 3. **Generate Text**
```bash
python inference.py --prompt "The future of AI is"
```

## 🔍 Advanced Usage

### **With Profiling**
```bash
# Train with detailed performance analysis
python examples/train_with_profiler.py

# Test profiler functionality
python tests/test_profiler.py
```

### **Manual Configuration**
```bash
# See auto-detected configuration
python auto_config.py

# Train with manual settings
python llm.py
```

## 📊 What Each Folder Contains

### **Core Files** (Root Directory)
- **Essential scripts** that users run directly
- **Main model implementation** (`llm.py`)
- **Auto-configuration system** (`auto_config.py`)
- **Setup and monitoring utilities**

### **profiler/** 
- **Advanced GPU Profiler** - comprehensive performance monitoring
- **Clean integration system** - optional profiling without modifying core code
- **Memory, kernel, data movement, and expert routing tracking**

### **tests/**
- **Test suite** for validating functionality
- **Profiler tests** - ensure profiling accuracy
- **Integration tests** - verify clean integration

### **examples/**
- **Example scripts** showing advanced usage
- **Profiling integration examples** - clean ways to add profiling
- **Best practices** for different use cases

### **docs/**
- **Comprehensive documentation** for advanced features
- **Profiler documentation** - detailed usage guide
- **API references** and examples

## 🚀 Development Workflow

### **For Users**
1. Run `./setup.sh` to install dependencies
2. Run `python train_auto.py` to train a model
3. Run `python inference.py` to generate text

### **For Developers**
1. **Core development**: Modify files in root directory
2. **Profiler development**: Work in `profiler/` folder
3. **Testing**: Add tests to `tests/` folder
4. **Examples**: Add examples to `examples/` folder
5. **Documentation**: Update files in `docs/` folder

### **For Contributors**
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** in appropriate folders
4. **Add tests** in `tests/` folder
5. **Update documentation** in `docs/` folder
6. **Submit pull request**

## 🔧 Import Structure

### **Core Imports**
```python
from llm import MoEMinimalLLM, MoEModelConfig
from auto_config import auto_configure
```

### **Profiler Imports**
```python
from profiler import AdvancedGPUProfiler, ProfilerContext
from profiler import profile_operation, set_profiler_hooks
```

### **Test Imports**
```python
# Tests automatically add the project root to Python path
from profiler import AdvancedGPUProfiler
```

## 📝 File Naming Conventions

- **Core files**: `snake_case.py` (e.g., `train_auto.py`)
- **Classes**: `PascalCase` (e.g., `MoEMinimalLLM`)
- **Functions**: `snake_case` (e.g., `train_moe_model`)
- **Constants**: `UPPER_CASE` (e.g., `MAX_STEPS`)
- **Packages**: `lowercase` (e.g., `profiler/`)

## 🎉 Benefits of This Structure

1. **Clean separation** - Core functionality vs. advanced features
2. **Easy navigation** - Logical folder organization
3. **Modular design** - Profiler is completely optional
4. **Scalable** - Easy to add new features and examples
5. **Professional** - Industry-standard project structure
6. **Maintainable** - Clear responsibilities for each folder

---

**Happy coding!** 🫐⚡📊
