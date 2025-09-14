#!/usr/bin/env python3
"""
Simple GPU Profiler for Blueberry LLM
Core measurements only: memory, kernel times, expert routing
"""

import time
import torch
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os

@dataclass
class MemoryStats:
    timestamp: float
    allocated_mb: float
    peak_mb: float
    expert_id: Optional[int] = None
    operation: str = ""

@dataclass
class KernelStats:
    timestamp: float
    kernel_name: str
    execution_time_ms: float
    expert_id: Optional[int] = None
    operation_type: str = ""

@dataclass
class ExpertStats:
    timestamp: float
    expert_indices: List[int]
    token_count: int
    load_balance_score: float

class SimpleGPUProfiler:
    """Simple GPU profiler with core measurements only"""
    
    def __init__(self, num_experts: int = 8, enable_profiling: bool = True):
        self.num_experts = num_experts
        self.enable_profiling = enable_profiling
        self.is_profiling = False
        self.start_time = 0.0
        
        # Core data storage
        self.memory_history: List[MemoryStats] = []
        self.kernel_history: List[KernelStats] = []
        self.expert_history: List[ExpertStats] = []
        
        # Aggregated stats
        self.kernel_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.expert_usage: Dict[int, int] = defaultdict(int)
        self.total_tokens = 0
        
    def start_profiling(self):
        """Start profiling session"""
        if not self.enable_profiling:
            return
        
        self.is_profiling = True
        self.start_time = time.time()
        print("🚀 Simple GPU Profiler started")
    
    def stop_profiling(self):
        """Stop profiling session"""
        if not self.enable_profiling:
            return
        
        self.is_profiling = False
        print("🛑 Simple GPU Profiler stopped")
        
        # Generate simple report
        self.generate_report()
    
    def profile_memory_allocation(self, size_bytes: int, expert_id: Optional[int] = None, operation: str = ""):
        """Profile memory allocation"""
        if not self.is_profiling or not torch.cuda.is_available():
            return
        
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        stats = MemoryStats(
            timestamp=time.time() - self.start_time,
            allocated_mb=allocated_mb,
            peak_mb=peak_mb,
            expert_id=expert_id,
            operation=operation
        )
        
        self.memory_history.append(stats)
    
    def profile_kernel_execution(self, kernel_name: str, expert_id: Optional[int] = None, operation_type: str = ""):
        """Profile kernel execution time"""
        if not self.is_profiling or not torch.cuda.is_available():
            return 0.0
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        # Kernel execution happens here
        end_event.record()
        torch.cuda.synchronize()
        
        execution_time_ms = start_event.elapsed_time(end_event)
        
        stats = KernelStats(
            timestamp=time.time() - self.start_time,
            kernel_name=kernel_name,
            execution_time_ms=execution_time_ms,
            expert_id=expert_id,
            operation_type=operation_type
        )
        
        self.kernel_history.append(stats)
        self.kernel_times[kernel_name].append(execution_time_ms)
        
        return execution_time_ms
    
    def profile_expert_routing(self, expert_indices: List[int], token_count: int):
        """Profile expert routing and load balancing"""
        if not self.is_profiling:
            return
        
        # Track expert usage
        for expert_id in expert_indices:
            self.expert_usage[expert_id] += token_count
        self.total_tokens += token_count
        
        # Calculate load balance score (lower is better)
        if len(expert_indices) > 0:
            avg_tokens = sum(self.expert_usage.values()) / len(self.expert_usage)
            max_tokens = max(self.expert_usage.values()) if self.expert_usage else 0
            load_balance_score = (max_tokens - avg_tokens) / max(avg_tokens, 1)
        else:
            load_balance_score = 0.0
        
        stats = ExpertStats(
            timestamp=time.time() - self.start_time,
            expert_indices=expert_indices,
            token_count=token_count,
            load_balance_score=load_balance_score
        )
        
        self.expert_history.append(stats)
    
    def print_dashboard(self):
        """Print simple profiling dashboard"""
        if not self.is_profiling:
            return
        
        print("\n🫐 Simple GPU Profiler Dashboard")
        print("=" * 40)
        
        # Memory stats
        if self.memory_history:
            latest_mem = self.memory_history[-1]
            print(f"📊 Memory:")
            print(f"  Current: {latest_mem.allocated_mb:.1f}MB")
            print(f"  Peak: {latest_mem.peak_mb:.1f}MB")
        
        # Kernel stats
        if self.kernel_times:
            print(f"\n⚡ Kernels:")
            for kernel_name, times in self.kernel_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    print(f"  {kernel_name}: {avg_time:.1f}ms avg")
        
        # Expert routing stats
        if self.expert_usage:
            print(f"\n🧠 Expert Routing:")
            total_usage = sum(self.expert_usage.values())
            for expert_id in sorted(self.expert_usage.keys()):
                usage = self.expert_usage[expert_id]
                percentage = (usage / max(total_usage, 1)) * 100
                print(f"  Expert {expert_id}: {percentage:.0f}%")
            
            # Load balance score
            if self.expert_history:
                latest_expert = self.expert_history[-1]
                print(f"  Load Balance Score: {latest_expert.load_balance_score:.2f}")
        
        print("=" * 40)
    
    def generate_report(self):
        """Generate simple JSON report"""
        if not self.enable_profiling:
            return
        
        # Create output directory
        os.makedirs("profiler_output", exist_ok=True)
        
        # Prepare report data
        report = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "profiling_duration": time.time() - self.start_time if self.start_time else 0,
            "num_experts": self.num_experts,
            "stats": {
                "memory": {
                    "peak_memory_mb": max([m.peak_mb for m in self.memory_history]) if self.memory_history else 0,
                    "current_allocated_mb": self.memory_history[-1].allocated_mb if self.memory_history else 0,
                },
                "kernel": {
                    "average_kernel_times": {
                        name: sum(times) / len(times) if times else 0
                        for name, times in self.kernel_times.items()
                    }
                },
                "expert_routing": {
                    "expert_utilization": [
                        (self.expert_usage[i] / max(sum(self.expert_usage.values()), 1)) * 100
                        for i in range(self.num_experts)
                    ],
                    "load_balance_score": self.expert_history[-1].load_balance_score if self.expert_history else 0,
                    "total_tokens": self.total_tokens
                }
            }
        }
        
        # Save report
        filename = f"profiler_output/simple_report_{report['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Simple report saved: {filename}")
    
    def get_current_stats(self):
        """Get current profiling statistics"""
        return {
            "memory_history": len(self.memory_history),
            "kernel_history": len(self.kernel_history),
            "expert_history": len(self.expert_history),
            "is_profiling": self.is_profiling
        }
