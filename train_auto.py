#!/usr/bin/env python3
"""
Auto-configured training for Blueberry LLM
Just run: python train_auto.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, random_split
from auto_config import auto_configure
from llm import train_moe_model, load_and_cache_data, TextTokenDataset

def auto_launch_distributed():
    """Auto-launch with torchrun if multi-GPU detected and not already in distributed mode"""
    import sys
    import subprocess
    
    # Check if we're already in distributed mode
    if 'RANK' in os.environ:
        return False  # Already launched with torchrun
    
    # Quick GPU check
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"🚀 Auto-launching with {num_gpus} GPUs...")
        
        # Re-launch with torchrun
        cmd = [
            'torchrun', 
            f'--nproc_per_node={num_gpus}',
            '--standalone',
            sys.argv[0]  # This script
        ] + sys.argv[1:]  # Plus any additional args
        
        try:
            result = subprocess.run(cmd, check=True)
            sys.exit(result.returncode)
        except subprocess.CalledProcessError as e:
            print(f"❌ Auto-launch failed: {e}")
            print(f"💡 Try manually: torchrun --nproc_per_node={num_gpus} train_auto.py")
            sys.exit(1)
    
    return False

def main():
    print("🫐 Starting Blueberry LLM Auto-Training")
    
    # Auto-launch with torchrun if needed
    auto_launch_distributed()
    
    # Auto-configure everything
    configurator = auto_configure()
    configurator.print_config()
    
    # Setup distributed training if needed
    if configurator.config.use_distributed:
        import torch.distributed as dist
        try:
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            print(f"🌐 Initialized data parallel training: rank {dist.get_rank()}/{dist.get_world_size()}")
            print(f"   Each GPU gets different data batches, same model")
        except Exception as e:
            print(f"❌ Distributed training failed: {e}")
            raise RuntimeError(f"Multi-GPU setup detected but distributed training failed: {e}")
    
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
    
    # Create data loaders with distributed sampler if needed
    train_sampler = None
    if configurator.config.use_distributed:
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler
        if dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"   Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Setup profiler if enabled
    profiler = None
    if configurator.config.enable_profiling:
        from profiler import SimpleGPUProfiler
        profiler = SimpleGPUProfiler(
            num_experts=model_config.num_experts,
            enable_profiling=True
        )
        # Add configuration attributes to profiler
        profiler.start_step = configurator.config.profiler_start_step
        profiler.end_step = configurator.config.profiler_end_step
        profiler.sample_rate = configurator.config.profiler_sample_rate
        print(f"📊 Profiler configured: {profiler.sample_rate*100:.0f}% sampling from step {profiler.start_step} to {profiler.end_step}")
    
    # Train the model
    print("\n🚀 Starting training...")
    model, final_metrics = train_moe_model(model_config, train_loader, val_loader, profiler)
    
    # Save results
    print("\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'auto_config': configurator.config,
        'tokenizer': tokenizer,
        'final_metrics': final_metrics
    }, 'blueberry_model.pt')
    
    print("✅ Training complete!")
    print(f"   Final validation loss: {final_metrics['val_loss']:.4f}")
    print(f"   Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Model saved as: blueberry_model.pt")

if __name__ == "__main__":
    main()
