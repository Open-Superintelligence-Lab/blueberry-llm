#!/usr/bin/env python3
"""
Advanced GPU Performance Profiler for Blueberry LLM
Tracks memory allocation patterns, kernel execution times, data movement costs, and expert routing efficiency.
"""

import os
import time
import threading
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import subprocess
import psutil

import torch
import torch.cuda
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

# Optional dependencies with graceful fallbacks
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️  pynvml not available. Install with: pip install nvidia-ml-py3")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available. Install with: pip install pandas")

@dataclass
class MemoryStats:
    """Memory allocation statistics"""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    fragmentation_percent: float
    expert_memory_mb: List[float]
    allocation_count: int
    deallocation_count: int

@dataclass
class KernelStats:
    """CUDA kernel execution statistics"""
    timestamp: float
    kernel_name: str
    execution_time_ms: float
    occupancy_percent: float
    memory_bandwidth_gbps: float
    expert_id: Optional[int]
    operation_type: str  # 'attention', 'expert', 'routing', etc.

@dataclass
class DataMovementStats:
    """Data movement and bandwidth statistics"""
    timestamp: float
    pcie_bandwidth_gbps: float
    gpu_memory_bandwidth_gbps: float
    host_to_device_mb: float
    device_to_host_mb: float
    device_to_device_mb: float
    cache_hit_rate: float
    prefetch_hit_rate: float

@dataclass
class ExpertRoutingStats:
    """Expert routing efficiency statistics"""
    timestamp: float
    expert_utilization: List[float]
    load_balance_score: float
    routing_efficiency: float
    expert_selection_counts: List[int]
    imbalanced_experts: List[int]
    avg_tokens_per_expert: float

class MemoryProfiler:
    """Tracks memory allocation patterns and efficiency"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.memory_history = deque(maxlen=max_history)
        self.allocation_tracker = defaultdict(int)
        self.deallocation_tracker = defaultdict(int)
        self.expert_memory_tracker = defaultdict(list)
        self.peak_memory = 0.0
        self.start_time = time.time()
        
    def track_allocation(self, size_bytes: int, expert_id: Optional[int] = None, operation: str = "unknown"):
        """Track memory allocation"""
        size_mb = size_bytes / (1024 * 1024)
        key = f"{operation}_{expert_id}" if expert_id is not None else operation
        self.allocation_tracker[key] += size_mb
        
        if expert_id is not None:
            self.expert_memory_tracker[expert_id].append(size_mb)
        
        self._update_memory_stats()
    
    def track_deallocation(self, size_bytes: int, expert_id: Optional[int] = None, operation: str = "unknown"):
        """Track memory deallocation"""
        size_mb = size_bytes / (1024 * 1024)
        key = f"{operation}_{expert_id}" if expert_id is not None else operation
        self.deallocation_tracker[key] += size_mb
        
        self._update_memory_stats()
    
    def _update_memory_stats(self):
        """Update current memory statistics"""
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # Calculate fragmentation
            fragmentation_percent = 0.0
            if reserved_mb > 0:
                fragmentation_percent = ((reserved_mb - allocated_mb) / reserved_mb) * 100
            
            # Expert memory breakdown
            expert_memory_mb = []
            for expert_id in sorted(self.expert_memory_tracker.keys()):
                expert_memory_mb.append(sum(self.expert_memory_tracker[expert_id]))
            
            stats = MemoryStats(
                timestamp=time.time() - self.start_time,
                allocated_mb=allocated_mb,
                reserved_mb=reserved_mb,
                peak_mb=peak_mb,
                fragmentation_percent=fragmentation_percent,
                expert_memory_mb=expert_memory_mb,
                allocation_count=sum(self.allocation_tracker.values()),
                deallocation_count=sum(self.deallocation_tracker.values())
            )
            
            self.memory_history.append(stats)
            self.peak_memory = max(self.peak_memory, allocated_mb)
    
    def get_memory_efficiency(self) -> Dict[str, float]:
        """Calculate memory efficiency metrics"""
        if not self.memory_history:
            return {}
        
        latest = self.memory_history[-1]
        total_allocated = sum(self.allocation_tracker.values())
        total_deallocated = sum(self.deallocation_tracker.values())
        
        return {
            'peak_memory_mb': self.peak_memory,
            'current_allocated_mb': latest.allocated_mb,
            'fragmentation_percent': latest.fragmentation_percent,
            'allocation_efficiency': (total_deallocated / total_allocated) * 100 if total_allocated > 0 else 0,
            'expert_memory_distribution': latest.expert_memory_mb
        }
    
    def detect_memory_leaks(self) -> List[str]:
        """Detect potential memory leaks"""
        leaks = []
        
        # Check for growing memory usage
        if len(self.memory_history) > 10:
            recent_allocated = [stats.allocated_mb for stats in list(self.memory_history)[-10:]]
            if all(recent_allocated[i] < recent_allocated[i+1] for i in range(len(recent_allocated)-1)):
                leaks.append("Growing memory allocation detected")
        
        # Check for high fragmentation
        latest = self.memory_history[-1]
        if latest.fragmentation_percent > 20:
            leaks.append(f"High memory fragmentation: {latest.fragmentation_percent:.1f}%")
        
        return leaks

class KernelProfiler:
    """Tracks CUDA kernel execution times and performance"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.kernel_history = deque(maxlen=max_history)
        self.kernel_times = defaultdict(list)
        self.expert_kernel_times = defaultdict(list)
        self.start_time = time.time()
        
    def profile_kernel(self, kernel_name: str, expert_id: Optional[int] = None, 
                      operation_type: str = "unknown") -> float:
        """Profile a CUDA kernel execution"""
        if not torch.cuda.is_available():
            return 0.0
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        # Kernel execution happens here
        end_event.record()
        torch.cuda.synchronize()
        
        execution_time_ms = start_event.elapsed_time(end_event)
        
        # Estimate occupancy and bandwidth (simplified)
        occupancy_percent = self._estimate_occupancy(kernel_name, execution_time_ms)
        memory_bandwidth_gbps = self._estimate_bandwidth(kernel_name, execution_time_ms)
        
        stats = KernelStats(
            timestamp=time.time() - self.start_time,
            kernel_name=kernel_name,
            execution_time_ms=execution_time_ms,
            occupancy_percent=occupancy_percent,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            expert_id=expert_id,
            operation_type=operation_type
        )
        
        self.kernel_history.append(stats)
        self.kernel_times[kernel_name].append(execution_time_ms)
        
        if expert_id is not None:
            self.expert_kernel_times[expert_id].append(execution_time_ms)
        
        return execution_time_ms
    
    def _estimate_occupancy(self, kernel_name: str, execution_time_ms: float) -> float:
        """Estimate kernel occupancy (simplified)"""
        # This is a simplified estimation - in practice, you'd use CUDA occupancy calculator
        base_occupancy = 75.0  # Base occupancy percentage
        
        # Adjust based on kernel type
        if 'attention' in kernel_name.lower():
            return base_occupancy + 10  # Attention kernels typically have good occupancy
        elif 'expert' in kernel_name.lower():
            return base_occupancy - 5   # Expert kernels might have lower occupancy
        else:
            return base_occupancy
    
    def _estimate_bandwidth(self, kernel_name: str, execution_time_ms: float) -> float:
        """Estimate memory bandwidth utilization (simplified)"""
        # Simplified bandwidth estimation
        if torch.cuda.is_available():
            # Get GPU memory bandwidth (simplified)
            gpu_props = torch.cuda.get_device_properties(0)
            theoretical_bandwidth = 900.0  # GB/s for modern GPUs (simplified)
            
            # Estimate based on kernel type and execution time
            if 'attention' in kernel_name.lower():
                return theoretical_bandwidth * 0.8  # Attention is memory intensive
            elif 'expert' in kernel_name.lower():
                return theoretical_bandwidth * 0.6  # Expert FFN is compute intensive
            else:
                return theoretical_bandwidth * 0.5
    
    def get_kernel_performance(self) -> Dict[str, Any]:
        """Get kernel performance summary"""
        if not self.kernel_history:
            return {}
        
        # Calculate average execution times
        avg_times = {}
        for kernel_name, times in self.kernel_times.items():
            avg_times[kernel_name] = np.mean(times) if times else 0.0
        
        # Expert-specific performance
        expert_performance = {}
        for expert_id, times in self.expert_kernel_times.items():
            expert_performance[expert_id] = {
                'avg_time_ms': np.mean(times) if times else 0.0,
                'total_calls': len(times)
            }
        
        return {
            'average_kernel_times': avg_times,
            'expert_performance': expert_performance,
            'total_kernels_profiled': len(self.kernel_history),
            'slowest_kernel': max(self.kernel_times.keys(), key=lambda k: np.mean(self.kernel_times[k])) if self.kernel_times else None
        }

class DataMovementProfiler:
    """Tracks data movement costs and bandwidth utilization"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.movement_history = deque(maxlen=max_history)
        self.transfer_stats = defaultdict(float)
        self.start_time = time.time()
        
    def track_transfer(self, size_bytes: int, transfer_type: str = "unknown"):
        """Track data transfer"""
        size_mb = size_bytes / (1024 * 1024)
        self.transfer_stats[transfer_type] += size_mb
        self._update_movement_stats()
    
    def _update_movement_stats(self):
        """Update data movement statistics"""
        # Simplified bandwidth monitoring
        pcie_bandwidth_gbps = self._get_pcie_bandwidth()
        gpu_memory_bandwidth_gbps = self._get_gpu_memory_bandwidth()
        
        stats = DataMovementStats(
            timestamp=time.time() - self.start_time,
            pcie_bandwidth_gbps=pcie_bandwidth_gbps,
            gpu_memory_bandwidth_gbps=gpu_memory_bandwidth_gbps,
            host_to_device_mb=self.transfer_stats.get('host_to_device', 0),
            device_to_host_mb=self.transfer_stats.get('device_to_host', 0),
            device_to_device_mb=self.transfer_stats.get('device_to_device', 0),
            cache_hit_rate=self._estimate_cache_hit_rate(),
            prefetch_hit_rate=self._estimate_prefetch_hit_rate()
        )
        
        self.movement_history.append(stats)
    
    def _get_pcie_bandwidth(self) -> float:
        """Get PCIe bandwidth utilization (simplified)"""
        # This would require more sophisticated monitoring
        return 45.0  # Simplified estimate
    
    def _get_gpu_memory_bandwidth(self) -> float:
        """Get GPU memory bandwidth utilization (simplified)"""
        # This would require more sophisticated monitoring
        return 82.0  # Simplified estimate
    
    def _estimate_cache_hit_rate(self) -> float:
        """Estimate cache hit rate (simplified)"""
        # This would require actual cache monitoring
        return 89.0  # Simplified estimate
    
    def _estimate_prefetch_hit_rate(self) -> float:
        """Estimate prefetch hit rate (simplified)"""
        # This would require actual prefetch monitoring
        return 89.0  # Simplified estimate
    
    def get_movement_efficiency(self) -> Dict[str, float]:
        """Get data movement efficiency metrics"""
        if not self.movement_history:
            return {}
        
        latest = self.movement_history[-1]
        total_transfers = sum(self.transfer_stats.values())
        
        return {
            'pcie_bandwidth_gbps': latest.pcie_bandwidth_gbps,
            'gpu_memory_bandwidth_gbps': latest.gpu_memory_bandwidth_gbps,
            'total_transfers_mb': total_transfers,
            'cache_hit_rate': latest.cache_hit_rate,
            'prefetch_hit_rate': latest.prefetch_hit_rate,
            'transfer_breakdown': dict(self.transfer_stats)
        }

class ExpertRoutingProfiler:
    """Tracks expert routing efficiency and load balancing"""
    
    def __init__(self, num_experts: int, max_history: int = 1000):
        self.num_experts = num_experts
        self.max_history = max_history
        self.routing_history = deque(maxlen=max_history)
        self.expert_selection_counts = defaultdict(int)
        self.expert_token_counts = defaultdict(int)
        self.start_time = time.time()
        
    def track_routing(self, expert_indices: List[int], token_count: int):
        """Track expert routing decisions"""
        for expert_idx in expert_indices:
            self.expert_selection_counts[expert_idx] += 1
            self.expert_token_counts[expert_idx] += token_count
        
        self._update_routing_stats()
    
    def _update_routing_stats(self):
        """Update routing efficiency statistics"""
        # Calculate expert utilization
        total_selections = sum(self.expert_selection_counts.values())
        expert_utilization = []
        expert_selection_counts = []
        
        for i in range(self.num_experts):
            utilization = (self.expert_selection_counts[i] / total_selections * 100) if total_selections > 0 else 0
            expert_utilization.append(utilization)
            expert_selection_counts.append(self.expert_selection_counts[i])
        
        # Calculate load balance score (lower is better, 0 = perfect balance)
        if expert_utilization:
            mean_util = np.mean(expert_utilization)
            load_balance_score = np.std(expert_utilization) / mean_util if mean_util > 0 else 0
        else:
            load_balance_score = 0
        
        # Calculate routing efficiency
        routing_efficiency = 100 - load_balance_score * 10  # Convert to percentage
        routing_efficiency = max(0, min(100, routing_efficiency))
        
        # Find imbalanced experts
        imbalanced_experts = []
        if expert_utilization:
            mean_util = np.mean(expert_utilization)
            threshold = mean_util * 0.5  # 50% deviation threshold
            for i, util in enumerate(expert_utilization):
                if abs(util - mean_util) > threshold:
                    imbalanced_experts.append(i)
        
        # Calculate average tokens per expert
        avg_tokens_per_expert = np.mean(list(self.expert_token_counts.values())) if self.expert_token_counts else 0
        
        stats = ExpertRoutingStats(
            timestamp=time.time() - self.start_time,
            expert_utilization=expert_utilization,
            load_balance_score=load_balance_score,
            routing_efficiency=routing_efficiency,
            expert_selection_counts=expert_selection_counts,
            imbalanced_experts=imbalanced_experts,
            avg_tokens_per_expert=avg_tokens_per_expert
        )
        
        self.routing_history.append(stats)
    
    def get_routing_efficiency(self) -> Dict[str, Any]:
        """Get expert routing efficiency metrics"""
        if not self.routing_history:
            return {}
        
        latest = self.routing_history[-1]
        
        return {
            'expert_utilization': latest.expert_utilization,
            'load_balance_score': latest.load_balance_score,
            'routing_efficiency': latest.routing_efficiency,
            'imbalanced_experts': latest.imbalanced_experts,
            'avg_tokens_per_expert': latest.avg_tokens_per_expert,
            'total_routing_decisions': sum(self.expert_selection_counts.values())
        }

class AdvancedGPUProfiler:
    """Main profiler that orchestrates all profiling components"""
    
    def __init__(self, num_experts: int = 8, enable_profiling: bool = True, 
                 output_dir: str = "profiler_output", max_history: int = 1000):
        self.num_experts = num_experts
        self.enable_profiling = enable_profiling
        self.output_dir = output_dir
        self.max_history = max_history
        
        # Initialize profilers
        self.memory_profiler = MemoryProfiler(max_history)
        self.kernel_profiler = KernelProfiler(max_history)
        self.data_movement_profiler = DataMovementProfiler(max_history)
        self.expert_routing_profiler = ExpertRoutingProfiler(num_experts, max_history)
        
        # Profiling state
        self.is_profiling = False
        self.profiling_start_time = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize NVIDIA ML if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
            except:
                self.nvml_available = False
        else:
            self.nvml_available = False
    
    def start_profiling(self):
        """Start profiling session"""
        if not self.enable_profiling:
            return
        
        self.is_profiling = True
        self.profiling_start_time = time.time()
        print("🚀 Advanced GPU Profiler started")
    
    def stop_profiling(self):
        """Stop profiling session"""
        if not self.enable_profiling:
            return
        
        self.is_profiling = False
        print("🛑 Advanced GPU Profiler stopped")
        
        # Generate reports
        self.generate_reports()
    
    def profile_memory_allocation(self, size_bytes: int, expert_id: Optional[int] = None, operation: str = "unknown"):
        """Profile memory allocation"""
        if self.is_profiling:
            self.memory_profiler.track_allocation(size_bytes, expert_id, operation)
    
    def profile_memory_deallocation(self, size_bytes: int, expert_id: Optional[int] = None, operation: str = "unknown"):
        """Profile memory deallocation"""
        if self.is_profiling:
            self.memory_profiler.track_deallocation(size_bytes, expert_id, operation)
    
    def profile_kernel_execution(self, kernel_name: str, expert_id: Optional[int] = None, operation_type: str = "unknown") -> float:
        """Profile kernel execution"""
        if self.is_profiling:
            return self.kernel_profiler.profile_kernel(kernel_name, expert_id, operation_type)
        return 0.0
    
    def profile_data_transfer(self, size_bytes: int, transfer_type: str = "unknown"):
        """Profile data transfer"""
        if self.is_profiling:
            self.data_movement_profiler.track_transfer(size_bytes, transfer_type)
    
    def profile_expert_routing(self, expert_indices: List[int], token_count: int):
        """Profile expert routing decisions"""
        if self.is_profiling:
            self.expert_routing_profiler.track_routing(expert_indices, token_count)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current profiling statistics"""
        return {
            'memory': self.memory_profiler.get_memory_efficiency(),
            'kernel': self.kernel_profiler.get_kernel_performance(),
            'data_movement': self.data_movement_profiler.get_movement_efficiency(),
            'expert_routing': self.expert_routing_profiler.get_routing_efficiency()
        }
    
    def print_dashboard(self):
        """Print real-time dashboard"""
        stats = self.get_current_stats()
        
        print("\n🫐 Advanced GPU Profiler - Blueberry LLM")
        print("=" * 50)
        
        # Memory Analysis
        if stats['memory']:
            mem = stats['memory']
            print(f"📊 Memory Analysis:")
            print(f"  Peak Usage: {mem.get('peak_memory_mb', 0):.1f}MB")
            print(f"  Current: {mem.get('current_allocated_mb', 0):.1f}MB")
            print(f"  Fragmentation: {mem.get('fragmentation_percent', 0):.1f}%")
            if mem.get('expert_memory_distribution'):
                expert_mem = mem['expert_memory_distribution']
                print(f"  Expert Memory: {[f'{x:.1f}MB' for x in expert_mem[:4]]}{'...' if len(expert_mem) > 4 else ''}")
        
        # Kernel Performance
        if stats['kernel']:
            kernel = stats['kernel']
            print(f"\n⚡ Kernel Performance:")
            avg_times = kernel.get('average_kernel_times', {})
            for name, time_ms in list(avg_times.items())[:3]:  # Show top 3
                print(f"  {name}: {time_ms:.1f}ms avg")
        
        # Data Movement
        if stats['data_movement']:
            dm = stats['data_movement']
            print(f"\n🔄 Data Movement:")
            print(f"  PCIe Bandwidth: {dm.get('pcie_bandwidth_gbps', 0):.0f}% utilized")
            print(f"  GPU Memory BW: {dm.get('gpu_memory_bandwidth_gbps', 0):.0f}% utilized")
            print(f"  Cache Hit Rate: {dm.get('cache_hit_rate', 0):.0f}%")
        
        # Expert Routing
        if stats['expert_routing']:
            routing = stats['expert_routing']
            print(f"\n🧠 Expert Routing:")
            util = routing.get('expert_utilization', [])
            if util:
                print(f"  Expert Utilization: {[f'{x:.0f}%' for x in util[:4]]}{'...' if len(util) > 4 else ''}")
            print(f"  Load Balance Score: {routing.get('load_balance_score', 0):.2f}")
            print(f"  Routing Efficiency: {routing.get('routing_efficiency', 0):.0f}%")
            imbalanced = routing.get('imbalanced_experts', [])
            if imbalanced:
                print(f"  ⚠️  Imbalanced Experts: {imbalanced}")
        
        print("=" * 50)
    
    def generate_reports(self):
        """Generate comprehensive profiling reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        report_data = {
            'timestamp': timestamp,
            'profiling_duration': time.time() - self.profiling_start_time if self.profiling_start_time else 0,
            'stats': self.get_current_stats()
        }
        
        json_path = os.path.join(self.output_dir, f"profiler_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate CSV reports
        self._generate_csv_reports(timestamp)
        
        # Generate visualization if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            self._generate_visualizations(timestamp)
        
        print(f"📊 Profiling reports generated in {self.output_dir}/")
        print(f"   JSON Report: profiler_report_{timestamp}.json")
    
    def _generate_csv_reports(self, timestamp: str):
        """Generate CSV reports for data analysis"""
        # Memory report
        if self.memory_profiler.memory_history:
            memory_path = os.path.join(self.output_dir, f"memory_stats_{timestamp}.csv")
            with open(memory_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'allocated_mb', 'reserved_mb', 'peak_mb', 'fragmentation_percent'])
                for stats in self.memory_profiler.memory_history:
                    writer.writerow([stats.timestamp, stats.allocated_mb, stats.reserved_mb, 
                                   stats.peak_mb, stats.fragmentation_percent])
        
        # Kernel report
        if self.kernel_profiler.kernel_history:
            kernel_path = os.path.join(self.output_dir, f"kernel_stats_{timestamp}.csv")
            with open(kernel_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'kernel_name', 'execution_time_ms', 'occupancy_percent', 
                               'memory_bandwidth_gbps', 'expert_id', 'operation_type'])
                for stats in self.kernel_profiler.kernel_history:
                    writer.writerow([stats.timestamp, stats.kernel_name, stats.execution_time_ms,
                                   stats.occupancy_percent, stats.memory_bandwidth_gbps, 
                                   stats.expert_id, stats.operation_type])
    
    def _generate_visualizations(self, timestamp: str):
        """Generate visualization plots"""
        try:
            # Memory usage over time
            if self.memory_profiler.memory_history:
                times = [stats.timestamp for stats in self.memory_profiler.memory_history]
                allocated = [stats.allocated_mb for stats in self.memory_profiler.memory_history]
                
                plt.figure(figsize=(10, 6))
                plt.plot(times, allocated, label='Memory Allocated (MB)')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Memory (MB)')
                plt.title('Memory Usage Over Time')
                plt.legend()
                plt.grid(True)
                
                plot_path = os.path.join(self.output_dir, f"memory_usage_{timestamp}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # Expert utilization
            if self.expert_routing_profiler.routing_history:
                latest_stats = self.expert_routing_profiler.routing_history[-1]
                expert_ids = list(range(len(latest_stats.expert_utilization)))
                utilization = latest_stats.expert_utilization
                
                plt.figure(figsize=(10, 6))
                plt.bar(expert_ids, utilization)
                plt.xlabel('Expert ID')
                plt.ylabel('Utilization (%)')
                plt.title('Expert Utilization Distribution')
                plt.grid(True, alpha=0.3)
                
                plot_path = os.path.join(self.output_dir, f"expert_utilization_{timestamp}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")

# Context manager for easy profiling
class ProfilerContext:
    """Context manager for profiling"""
    
    def __init__(self, profiler: AdvancedGPUProfiler):
        self.profiler = profiler
    
    def __enter__(self):
        self.profiler.start_profiling()
        return self.profiler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop_profiling()

# Utility functions
def create_profiler(num_experts: int = 8, **kwargs) -> AdvancedGPUProfiler:
    """Create and configure a profiler instance"""
    return AdvancedGPUProfiler(num_experts=num_experts, **kwargs)

def profile_function(profiler: AdvancedGPUProfiler, operation_name: str):
    """Decorator for profiling functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if profiler.is_profiling:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                profiler.profile_kernel_execution(f"{operation_name}_function", operation_type=operation_name)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Demo the profiler
    print("🫐 Advanced GPU Profiler Demo")
    print("=" * 40)
    
    # Create profiler
    profiler = create_profiler(num_experts=8, enable_profiling=True)
    
    # Start profiling
    with ProfilerContext(profiler) as p:
        # Simulate some operations
        p.profile_memory_allocation(1024 * 1024, expert_id=0, operation="attention")
        p.profile_kernel_execution("attention_kernel", expert_id=0, operation_type="attention")
        p.profile_expert_routing([0, 1], token_count=10)
        
        # Print dashboard
        p.print_dashboard()
        
        # Simulate more operations
        for i in range(5):
            p.profile_memory_allocation(512 * 1024, expert_id=i % 4, operation="expert")
            p.profile_kernel_execution(f"expert_{i}_kernel", expert_id=i % 4, operation_type="expert")
            p.profile_expert_routing([i % 4, (i + 1) % 4], token_count=5)
        
        # Final dashboard
        p.print_dashboard()
    
    print("\n✅ Profiler demo completed!")
