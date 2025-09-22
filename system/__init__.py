"""
System configuration and GPU detection module.

This module provides a singleton SystemConfig class that detects and caches
GPU capabilities at startup, enabling efficient runtime dispatch to optimized kernels.
"""

import torch
from typing import Tuple, Dict, Any


class SystemConfig:
    """
    Singleton class that detects and caches GPU system capabilities.
    
    This class is initialized once at startup and provides information about:
    - Number of available GPUs
    - GPU compute capability (major, minor)
    - Support for specific features (FP8, Tensor Cores, etc.)
    - Memory information
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._detect_gpu_capabilities()
            SystemConfig._initialized = True
    
    def _detect_gpu_capabilities(self):
        """Detect and cache GPU capabilities."""
        self.device_count = torch.cuda.device_count()
        
        if self.device_count > 0:
            # Get compute capability of the first GPU (assuming homogeneous cluster)
            self.capability = torch.cuda.get_device_capability(0)
            self.device_name = torch.cuda.get_device_name(0)
            
            # Feature detection based on compute capability
            self.has_fp8_support = self.capability >= (9, 0)  # Blackwell+
            self.has_tensor_cores = self.capability >= (7, 0)  # Volta+
            self.has_bf16_support = self.capability >= (8, 0)  # Ampere+
            
            # Memory information
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            self.memory_gb = self.total_memory / (1024**3)
            
            # Architecture classification
            self.architecture = self._classify_architecture()
            
        else:
            # No CUDA available
            self.capability = (0, 0)
            self.device_name = "CPU"
            self.has_fp8_support = False
            self.has_tensor_cores = False
            self.has_bf16_support = False
            self.total_memory = 0
            self.memory_gb = 0
            self.architecture = "cpu"
    
    def _classify_architecture(self) -> str:
        """Classify GPU architecture based on compute capability."""
        major, minor = self.capability
        
        if major >= 9:
            return "blackwell"
        elif major == 8:
            if minor >= 9:  # H100
                return "hopper"
            else:  # A100, RTX 30xx
                return "ampere"
        elif major == 7:
            if minor >= 5:  # RTX 20xx
                return "turing"
            else:  # V100
                return "volta"
        elif major == 6:
            return "pascal"
        else:
            return "unknown"
    
    def get_optimal_dtype(self) -> torch.dtype:
        """Get the optimal data type for this GPU architecture."""
        if self.has_fp8_support:
            return torch.float8_e4m3fn
        elif self.has_bf16_support:
            return torch.bfloat16
        else:
            return torch.float16
    
    def supports_feature(self, feature: str) -> bool:
        """Check if the current GPU supports a specific feature."""
        feature_map = {
            "fp8": self.has_fp8_support,
            "tensor_cores": self.has_tensor_cores,
            "bf16": self.has_bf16_support,
        }
        return feature_map.get(feature, False)
    
    def get_info(self) -> Dict[str, Any]:
        """Get a dictionary with all system information."""
        return {
            "device_count": self.device_count,
            "capability": self.capability,
            "device_name": self.device_name,
            "architecture": self.architecture,
            "memory_gb": self.memory_gb,
            "has_fp8_support": self.has_fp8_support,
            "has_tensor_cores": self.has_tensor_cores,
            "has_bf16_support": self.has_bf16_support,
            "optimal_dtype": str(self.get_optimal_dtype()),
        }
    
    def __repr__(self) -> str:
        return f"SystemConfig(arch={self.architecture}, capability={self.capability}, gpus={self.device_count})"


# Create a single global instance
SYSTEM_CONFIG = SystemConfig()


def get_system_config() -> SystemConfig:
    """Get the global system configuration instance."""
    return SYSTEM_CONFIG


def print_system_info():
    """Print detailed system information for debugging."""
    config = get_system_config()
    info = config.get_info()
    
    print("🔍 GPU System Configuration:")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Device: {info['device_name']}")
    print(f"   Compute Capability: {info['capability']}")
    print(f"   GPU Count: {info['device_count']}")
    print(f"   Memory: {info['memory_gb']:.1f} GB")
    print(f"   FP8 Support: {info['has_fp8_support']}")
    print(f"   Tensor Cores: {info['has_tensor_cores']}")
    print(f"   BF16 Support: {info['has_bf16_support']}")
    print(f"   Optimal Dtype: {info['optimal_dtype']}")


if __name__ == "__main__":
    print_system_info()
