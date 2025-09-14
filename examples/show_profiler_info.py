#!/usr/bin/env python3
"""
Show profiler information from existing training runs
"""

import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler import AdvancedGPUProfiler

def show_profiler_info():
    """Show profiler information from saved data"""
    print("🔍 Blueberry LLM Profiler Information")
    print("=" * 50)
    
    # Check for profiler output directory
    profiler_dir = "profiler_output"
    if not os.path.exists(profiler_dir):
        print(f"❌ No profiler output found in {profiler_dir}/")
        print("💡 Run training with profiling first:")
        print("   python examples/train_with_live_profiler.py")
        return
    
    print(f"📁 Found profiler output in {profiler_dir}/")
    
    # List available reports
    files = os.listdir(profiler_dir)
    json_files = [f for f in files if f.endswith('.json')]
    csv_files = [f for f in files if f.endswith('.csv')]
    png_files = [f for f in files if f.endswith('.png')]
    
    print(f"\n📊 Available Reports:")
    print(f"   JSON Reports: {len(json_files)}")
    print(f"   CSV Reports: {len(csv_files)}")
    print(f"   Visualization Plots: {len(png_files)}")
    
    if json_files:
        print(f"\n📄 JSON Reports:")
        for f in sorted(json_files):
            print(f"   - {f}")
    
    if csv_files:
        print(f"\n📊 CSV Reports:")
        for f in sorted(csv_files):
            print(f"   - {f}")
    
    if png_files:
        print(f"\n📈 Visualization Plots:")
        for f in sorted(png_files):
            print(f"   - {f}")
    
    # Show latest JSON report
    if json_files:
        latest_json = sorted(json_files)[-1]
        json_path = os.path.join(profiler_dir, latest_json)
        
        print(f"\n📊 Latest Report: {latest_json}")
        print("=" * 50)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            print(f"📅 Timestamp: {data.get('timestamp', 'Unknown')}")
            print(f"⏱️  Profiling Duration: {data.get('profiling_duration', 0):.1f} seconds")
            
            stats = data.get('stats', {})
            
            # Memory stats
            if 'memory' in stats and stats['memory']:
                mem = stats['memory']
                print(f"\n📊 Memory Analysis:")
                print(f"   Peak Memory: {mem.get('peak_memory_mb', 0):.1f} MB")
                print(f"   Current Allocated: {mem.get('current_allocated_mb', 0):.1f} MB")
                print(f"   Fragmentation: {mem.get('fragmentation_percent', 0):.1f}%")
                if mem.get('expert_memory_distribution'):
                    expert_mem = mem['expert_memory_distribution']
                    print(f"   Expert Memory: {[f'{x:.1f}MB' for x in expert_mem[:4]]}{'...' if len(expert_mem) > 4 else ''}")
            
            # Kernel stats
            if 'kernel' in stats and stats['kernel']:
                kernel = stats['kernel']
                print(f"\n⚡ Kernel Performance:")
                avg_times = kernel.get('average_kernel_times', {})
                for name, time_ms in list(avg_times.items())[:3]:
                    print(f"   {name}: {time_ms:.1f}ms avg")
            
            # Expert routing stats
            if 'expert_routing' in stats and stats['expert_routing']:
                routing = stats['expert_routing']
                print(f"\n🧠 Expert Routing:")
                util = routing.get('expert_utilization', [])
                if util:
                    print(f"   Expert Utilization: {[f'{x:.0f}%' for x in util[:4]]}{'...' if len(util) > 4 else ''}")
                print(f"   Load Balance Score: {routing.get('load_balance_score', 0):.2f}")
                print(f"   Routing Efficiency: {routing.get('routing_efficiency', 0):.0f}%")
                imbalanced = routing.get('imbalanced_experts', [])
                if imbalanced:
                    print(f"   ⚠️  Imbalanced Experts: {imbalanced}")
            
            # Data movement stats
            if 'data_movement' in stats and stats['data_movement']:
                dm = stats['data_movement']
                print(f"\n🔄 Data Movement:")
                print(f"   PCIe Bandwidth: {dm.get('pcie_bandwidth_gbps', 0):.0f}% utilized")
                print(f"   GPU Memory BW: {dm.get('gpu_memory_bandwidth_gbps', 0):.0f}% utilized")
                print(f"   Cache Hit Rate: {dm.get('cache_hit_rate', 0):.0f}%")
            
        except Exception as e:
            print(f"❌ Error reading report: {e}")
    
    print(f"\n💡 To view visualizations:")
    print(f"   Open the PNG files in {profiler_dir}/ with an image viewer")
    print(f"\n💡 To analyze CSV data:")
    print(f"   Import CSV files into Excel, Python pandas, or other analysis tools")

def show_live_profiler_demo():
    """Show a demo of live profiler"""
    print(f"\n🎯 Live Profiler Demo")
    print("=" * 30)
    
    # Create a simple profiler demo
    profiler = AdvancedGPUProfiler(num_experts=4, enable_profiling=True)
    
    with profiler.start_profiling():
        print("🚀 Simulating MoE operations...")
        
        # Simulate some operations
        for i in range(5):
            # Simulate memory allocation
            size_bytes = (i + 1) * 1024 * 1024  # 1MB, 2MB, 3MB, 4MB, 5MB
            profiler.profile_memory_allocation(size_bytes, expert_id=i % 4, operation=f"demo_alloc_{i}")
            
            # Simulate kernel execution
            profiler.profile_kernel_execution(f"demo_kernel_{i}", expert_id=i % 4, operation_type="demo")
            
            # Simulate expert routing
            expert_indices = [i % 4, (i + 1) % 4]
            token_count = 16
            profiler.profile_expert_routing(expert_indices, token_count)
        
        # Show dashboard
        print(f"\n📊 LIVE PROFILER DASHBOARD:")
        profiler.print_dashboard()
    
    profiler.stop_profiling()

if __name__ == "__main__":
    try:
        show_profiler_info()
        
        # Ask if user wants to see live demo
        print(f"\n🎯 Want to see a live profiler demo? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                show_live_profiler_demo()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
