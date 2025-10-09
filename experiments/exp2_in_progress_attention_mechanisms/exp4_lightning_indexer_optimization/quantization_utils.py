"""
Quantization Utilities for Experiment 4

This module provides utilities for quantizing Lightning Indexer computations
to reduce memory usage and improve speed while maintaining quality.

Quantization strategies:
1. FP16 (half-precision) for indexer computations
2. Mixed precision (FP32 main attention, FP16 indexer)
3. INT8 quantization for indexer weights
4. Dynamic quantization based on activation ranges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import warnings


class FP16IndexerWrapper(nn.Module):
    """
    Wrapper that applies FP16 quantization to indexer computations
    
    Args:
        indexer: Lightning Indexer to wrap
        use_autocast: Use torch.cuda.amp.autocast for automatic mixed precision
    """
    def __init__(self, indexer: nn.Module, use_autocast: bool = True):
        super().__init__()
        self.indexer = indexer
        self.use_autocast = use_autocast
        
        # Store original dtype
        self.original_dtype = next(indexer.parameters()).dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP16 quantization"""
        if self.use_autocast and x.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return self._forward_fp16(x)
        else:
            return self._forward_fp16(x)
    
    def _forward_fp16(self, x: torch.Tensor) -> torch.Tensor:
        """Manual FP16 forward pass"""
        # Convert input to FP16
        x_fp16 = x.half() if x.dtype == torch.float32 else x
        
        # Forward through indexer in FP16
        with torch.cuda.amp.autocast(enabled=False):
            # Temporarily convert indexer to FP16
            self.indexer.half()
            index_scores_fp16 = self.indexer(x_fp16)
            # Convert back to original dtype
            self.indexer.to(self.original_dtype)
        
        # Convert output back to original dtype
        if x.dtype == torch.float32:
            return index_scores_fp16.float()
        else:
            return index_scores_fp16


class MixedPrecisionIndexer(nn.Module):
    """
    Mixed precision indexer: FP32 for main computations, FP16 for indexer
    
    Args:
        indexer: Lightning Indexer to wrap
        main_dtype: Dtype for main attention computations
        indexer_dtype: Dtype for indexer computations
    """
    def __init__(
        self, 
        indexer: nn.Module, 
        main_dtype: torch.dtype = torch.float32,
        indexer_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.indexer = indexer
        self.main_dtype = main_dtype
        self.indexer_dtype = indexer_dtype
        
        # Store original dtype
        self.original_dtype = next(indexer.parameters()).dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision"""
        # Convert input to indexer dtype
        x_indexer = x.to(self.indexer_dtype)
        
        # Forward through indexer in indexer dtype
        self.indexer.to(self.indexer_dtype)
        index_scores = self.indexer(x_indexer)
        
        # Convert output back to main dtype
        index_scores = index_scores.to(self.main_dtype)
        
        # Convert indexer back to original dtype
        self.indexer.to(self.original_dtype)
        
        return index_scores


class DynamicQuantizedIndexer(nn.Module):
    """
    Dynamically quantized indexer that adapts quantization based on activation ranges
    
    Args:
        indexer: Lightning Indexer to wrap
        quantization_bits: Number of bits for quantization (4, 8, or 16)
        calibration_steps: Number of steps to calibrate quantization parameters
    """
    def __init__(
        self, 
        indexer: nn.Module, 
        quantization_bits: int = 8,
        calibration_steps: int = 100
    ):
        super().__init__()
        self.indexer = indexer
        self.quantization_bits = quantization_bits
        self.calibration_steps = calibration_steps
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_step = 0
        self.activation_ranges = {}
        
        # Quantization parameters
        self.scales = {}
        self.zero_points = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic quantization"""
        if not self.is_calibrated and self.training:
            return self._calibrate_and_forward(x)
        else:
            return self._quantized_forward(x)
    
    def _calibrate_and_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrate quantization parameters and forward pass"""
        # Forward through indexer normally
        index_scores = self.indexer(x)
        
        # Collect activation statistics
        self._collect_activation_stats(x, 'input')
        self._collect_activation_stats(index_scores, 'output')
        
        self.calibration_step += 1
        
        # Check if calibration is complete
        if self.calibration_step >= self.calibration_steps:
            self._finalize_calibration()
        
        return index_scores
    
    def _quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization"""
        if self.quantization_bits == 16:
            # Use FP16 quantization
            x_quantized = x.half()
            self.indexer.half()
            index_scores = self.indexer(x_quantized)
            self.indexer.float()
            return index_scores.float()
        
        elif self.quantization_bits == 8:
            # Use INT8 quantization
            return self._int8_forward(x)
        
        elif self.quantization_bits == 4:
            # Use INT4 quantization
            return self._int4_forward(x)
        
        else:
            # No quantization
            return self.indexer(x)
    
    def _int8_forward(self, x: torch.Tensor) -> torch.Tensor:
        """INT8 quantized forward pass"""
        # Quantize input
        x_quantized, x_scale, x_zero_point = self._quantize_tensor(x, 'input')
        
        # Forward through indexer (simplified - in practice would need custom kernels)
        # For now, just apply scale adjustment
        x_dequantized = self._dequantize_tensor(x_quantized, x_scale, x_zero_point)
        index_scores = self.indexer(x_dequantized)
        
        # Quantize output
        scores_quantized, scores_scale, scores_zero_point = self._quantize_tensor(index_scores, 'output')
        scores_dequantized = self._dequantize_tensor(scores_quantized, scores_scale, scores_zero_point)
        
        return scores_dequantized
    
    def _int4_forward(self, x: torch.Tensor) -> torch.Tensor:
        """INT4 quantized forward pass (simplified)"""
        # For INT4, we'll use a simplified approach
        # In practice, this would require custom kernels
        warnings.warn("INT4 quantization not fully implemented, using FP16 instead")
        return self._quantized_forward(x.half()).float()
    
    def _collect_activation_stats(self, tensor: torch.Tensor, name: str):
        """Collect activation statistics for calibration"""
        if name not in self.activation_ranges:
            self.activation_ranges[name] = {
                'min': float('inf'),
                'max': float('-inf')
            }
        
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        self.activation_ranges[name]['min'] = min(
            self.activation_ranges[name]['min'], tensor_min
        )
        self.activation_ranges[name]['max'] = max(
            self.activation_ranges[name]['max'], tensor_max
        )
    
    def _finalize_calibration(self):
        """Finalize calibration and compute quantization parameters"""
        for name, ranges in self.activation_ranges.items():
            min_val = ranges['min']
            max_val = ranges['max']
            
            if self.quantization_bits == 8:
                # INT8: range [-128, 127]
                scale = (max_val - min_val) / 255.0
                zero_point = int(-min_val / scale) - 128
            elif self.quantization_bits == 4:
                # INT4: range [-8, 7]
                scale = (max_val - min_val) / 15.0
                zero_point = int(-min_val / scale) - 8
            else:
                continue
            
            self.scales[name] = scale
            self.zero_points[name] = zero_point
        
        self.is_calibrated = True
        print(f"Quantization calibration complete for {self.quantization_bits}-bit")
    
    def _quantize_tensor(self, tensor: torch.Tensor, name: str) -> Tuple[torch.Tensor, float, int]:
        """Quantize tensor using calibrated parameters"""
        if name not in self.scales:
            return tensor, 1.0, 0
        
        scale = self.scales[name]
        zero_point = self.zero_points[name]
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        
        # Clamp to valid range
        if self.quantization_bits == 8:
            quantized = torch.clamp(quantized, -128, 127)
        elif self.quantization_bits == 4:
            quantized = torch.clamp(quantized, -8, 7)
        
        return quantized.int(), scale, zero_point
    
    def _dequantize_tensor(self, quantized: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        """Dequantize tensor back to float"""
        return (quantized.float() - zero_point) * scale


class QuantizationBenchmark:
    """Utility class for benchmarking quantization performance"""
    
    @staticmethod
    def benchmark_indexer(
        indexer: nn.Module,
        x: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark indexer performance"""
        device = next(indexer.parameters()).device
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = indexer(x)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        
        if start_time is not None:
            start_time.record()
        else:
            import time
            start_time_val = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = indexer(x)
        
        if end_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / num_runs  # ms
        else:
            elapsed_time = (time.time() - start_time_val) * 1000 / num_runs  # ms
        
        # Memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
        else:
            memory_allocated = 0
        
        return {
            'avg_time_ms': elapsed_time,
            'memory_mb': memory_allocated,
            'throughput_tokens_per_sec': x.numel() / (elapsed_time / 1000)
        }
    
    @staticmethod
    def compare_quantization_strategies(
        original_indexer: nn.Module,
        x: torch.Tensor,
        strategies: list = ['fp16', 'mixed_precision', 'int8', 'int4']
    ) -> Dict[str, Dict[str, float]]:
        """Compare different quantization strategies"""
        results = {}
        
        # Original
        results['original'] = QuantizationBenchmark.benchmark_indexer(original_indexer, x)
        
        # FP16
        if 'fp16' in strategies:
            fp16_indexer = FP16IndexerWrapper(original_indexer)
            results['fp16'] = QuantizationBenchmark.benchmark_indexer(fp16_indexer, x)
        
        # Mixed precision
        if 'mixed_precision' in strategies:
            mixed_indexer = MixedPrecisionIndexer(original_indexer)
            results['mixed_precision'] = QuantizationBenchmark.benchmark_indexer(mixed_indexer, x)
        
        # Dynamic quantization
        if 'int8' in strategies:
            int8_indexer = DynamicQuantizedIndexer(original_indexer, quantization_bits=8)
            results['int8'] = QuantizationBenchmark.benchmark_indexer(int8_indexer, x)
        
        if 'int4' in strategies:
            int4_indexer = DynamicQuantizedIndexer(original_indexer, quantization_bits=4)
            results['int4'] = QuantizationBenchmark.benchmark_indexer(int4_indexer, x)
        
        return results
    
    @staticmethod
    def analyze_quantization_error(
        original_indexer: nn.Module,
        quantized_indexer: nn.Module,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """Analyze quantization error"""
        original_indexer.eval()
        quantized_indexer.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        max_error = 0.0
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Forward through both indexers
                original_output = original_indexer(x)
                quantized_output = quantized_indexer(x)
                
                # Compute errors
                mse = torch.mean((original_output - quantized_output) ** 2).item()
                mae = torch.mean(torch.abs(original_output - quantized_output)).item()
                max_err = torch.max(torch.abs(original_output - quantized_output)).item()
                
                total_mse += mse
                total_mae += mae
                max_error = max(max_error, max_err)
        
        return {
            'mse': total_mse / num_samples,
            'mae': total_mae / num_samples,
            'max_error': max_error,
            'rmse': (total_mse / num_samples) ** 0.5
        }


def create_quantized_indexer(
    indexer: nn.Module,
    quantization_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create quantized indexer variants
    
    Args:
        indexer: Original indexer to quantize
        quantization_type: Type of quantization ('fp16', 'mixed_precision', 'int8', 'int4')
        **kwargs: Additional arguments for quantization
    
    Returns:
        Quantized indexer instance
    """
    if quantization_type == 'fp16':
        return FP16IndexerWrapper(indexer, **kwargs)
    elif quantization_type == 'mixed_precision':
        return MixedPrecisionIndexer(indexer, **kwargs)
    elif quantization_type == 'int8':
        return DynamicQuantizedIndexer(indexer, quantization_bits=8, **kwargs)
    elif quantization_type == 'int4':
        return DynamicQuantizedIndexer(indexer, quantization_bits=4, **kwargs)
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
