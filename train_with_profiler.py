#!/usr/bin/env python3
"""
Training script with Advanced GPU Profiler integration
This demonstrates how to use the profiler without modifying the core model code.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, random_split
from auto_config import auto_configure
from llm import train_moe_model, load_and_cache_data, TextTokenDataset, MoEModelConfig
from advanced_gpu_profiler import AdvancedGPUProfiler
from profiler_hooks import ProfilerContext, patch_model_for_profiling

def train_with_profiling():
    """Train MoE model with comprehensive profiling"""
    print("🫐 Blueberry LLM Training with Advanced GPU Profiler")
    print("=" * 60)
    
    # Auto-configure everything
    configurator = auto_configure()
    configurator.print_config()
    
    # Get model configuration
    model_config = configurator.get_model_config()
    
    # Auto-size dataset based on hardware
    if configurator.config.num_gpus == 0:
        model_config.num_documents = 500
        model_config.max_tokens = 50000
    elif configurator.config.gpu_memory_gb < 16:
        model_config.num_documents = 1000
        model_config.max_tokens = 100000
    elif configurator.config.num_gpus <= 2:
        model_config.num_documents = 2000
        model_config.max_tokens = 250000
    else:
        model_config.num_documents = 5000
        model_config.max_tokens = 500000
    
    print(f"\n📊 Loading {model_config.num_documents} documents, {model_config.max_tokens:,} tokens...")
    
    # Load data
    texts, tokenizer, tokens = load_and_cache_data(model_config)
    dataset = TextTokenDataset(tokens, model_config.max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"   Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create profiler
    profiler = AdvancedGPUProfiler(
        num_experts=model_config.num_experts, 
        enable_profiling=True,
        output_dir="profiler_output"
    )
    
    print(f"\n🔍 Advanced GPU Profiler enabled")
    print(f"   Experts: {model_config.num_experts}")
    print(f"   Output directory: profiler_output/")
    
    # Train with profiling using context manager
    with ProfilerContext(profiler) as p:
        print(f"\n🚀 Starting training with profiling...")
        
        # Train the model (clean, no profiling code in core training)
        model, final_metrics = train_moe_model(model_config, train_loader, val_loader)
        
        # Print final profiling dashboard
        print(f"\n📊 Final Profiling Results:")
        p.print_dashboard()
    
    # Save model with profiling metadata
    print(f"\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'auto_config': configurator.config,
        'tokenizer': tokenizer,
        'final_metrics': final_metrics,
        'profiler_stats': profiler.get_current_stats()
    }, 'blueberry_model_with_profiling.pt')
    
    print("✅ Training with profiling complete!")
    print(f"   Final validation loss: {final_metrics['val_loss']:.4f}")
    print(f"   Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Model saved as: blueberry_model_with_profiling.pt")
    print(f"   Profiling reports saved in: profiler_output/")

def train_with_manual_profiling():
    """Example of manual profiling integration"""
    print("\n🔧 Manual Profiling Integration Example")
    print("=" * 50)
    
    # Create profiler
    profiler = AdvancedGPUProfiler(num_experts=8, enable_profiling=True)
    
    # Manual profiling control
    profiler.start_profiling()
    
    # Simulate some operations
    print("📊 Simulating MoE operations...")
    
    # Simulate memory allocations
    for i in range(5):
        size_bytes = (i + 1) * 1024 * 1024  # 1MB, 2MB, 3MB, 4MB, 5MB
        profiler.profile_memory_allocation(size_bytes, expert_id=i % 4, operation=f"simulated_alloc_{i}")
    
    # Simulate kernel executions
    for i in range(5):
        profiler.profile_kernel_execution(f"simulated_kernel_{i}", expert_id=i % 4, operation_type="simulation")
    
    # Simulate expert routing
    for i in range(10):
        expert_indices = [i % 4, (i + 1) % 4]  # Simulate top-2 routing
        token_count = 16
        profiler.profile_expert_routing(expert_indices, token_count)
    
    # Print dashboard
    profiler.print_dashboard()
    
    # Stop profiling
    profiler.stop_profiling()
    
    print("✅ Manual profiling example completed!")

def train_with_monkey_patching():
    """Example of automatic profiling with monkey patching"""
    print("\n🐒 Monkey Patching Profiling Example")
    print("=" * 50)
    
    from llm import MoEMinimalLLM, MoEModelConfig
    
    # Create a simple model
    config = MoEModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        num_experts=4,
        vocab_size=1000
    )
    
    model = MoEMinimalLLM(config)
    
    # Create profiler
    profiler = AdvancedGPUProfiler(num_experts=4, enable_profiling=True)
    
    # Patch model for automatic profiling
    patched_model = patch_model_for_profiling(model, profiler)
    
    print("📊 Model patched for automatic profiling")
    
    # Simulate forward passes
    with ProfilerContext(profiler) as p:
        print("🚀 Running forward passes...")
        
        # Simulate some forward passes
        for i in range(3):
            x = torch.randint(0, config.vocab_size, (2, 16))  # batch_size=2, seq_len=16
            with torch.no_grad():
                output = patched_model(x)
            print(f"  Forward pass {i+1}/3 completed")
        
        # Print profiling results
        p.print_dashboard()
    
    print("✅ Monkey patching example completed!")

if __name__ == "__main__":
    try:
        # Run main training with profiling
        train_with_profiling()
        
        # Run examples
        train_with_manual_profiling()
        train_with_monkey_patching()
        
        print("\n🎉 All profiling examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
