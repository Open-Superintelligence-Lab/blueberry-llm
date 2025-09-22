#!/usr/bin/env python3
"""
Auto-configured training for Blueberry LLM
Just run: python train_auto.py
"""

import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

import torch
import argparse
from torch.utils.data import DataLoader, random_split
from core.auto_config import auto_configure
from legacy.llm import train_moe_model, load_and_cache_data, TextTokenDataset

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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-configured training for Blueberry LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Megatron options
    parser.add_argument("--use-megatron", action="store_true", help="Force Megatron backend (enable Megatron)")
    parser.add_argument("--no-megatron", action="store_true", help="Force native backend (disable Megatron)")
    
    return parser.parse_args()

def main():
    print("🫐 Starting Blueberry LLM Auto-Training")
    
    # Parse arguments
    args = parse_arguments()
    
    # Auto-launch with torchrun if needed
    auto_launch_distributed()
    
    # Auto-configure everything
    configurator = auto_configure()
    
    # Override Megatron settings if specified
    if args.use_megatron:
        configurator.config.use_megatron = True
        print("🚀 Megatron forced enabled via --use-megatron flag")
    elif args.no_megatron:
        configurator.config.use_megatron = False
        print("🚀 Megatron forced disabled via --no-megatron flag")
    
    configurator.print_config()
    
    # Print detailed GPU system information
    print("\n🔍 Detailed GPU System Information:")
    print("=" * 50)
    from system import print_system_info
    print_system_info()
    print("=" * 50)
    
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
    
    # Auto-size dataset based on hardware (use already-detected config)
    if configurator.config.num_gpus == 0:
        model_config.num_documents = 500
        model_config.max_tokens = 50000
    elif configurator.config.gpu_memory_gb >= 14:  # T4 and similar high-memory GPUs
        model_config.num_documents = 2000
        model_config.max_tokens = 200000
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
    
    # Train the model
    print("\n🚀 Starting training...")
    
    # Use Megatron-enabled training if requested
    if configurator.config.use_megatron:
        print("🚀 Using Megatron-enabled training pipeline...")
        from models import create_model
        from training import train_model
        from configs import AdaptiveMoEModelConfig
        
        # Convert legacy config to new config format
        adaptive_config = AdaptiveMoEModelConfig(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            d_ff=model_config.d_ff,
            batch_size=model_config.batch_size,
            max_steps=model_config.max_steps,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            muon_lr=model_config.muon_lr,
            max_seq_len=model_config.max_seq_len,
            num_experts=model_config.num_experts,
            use_amp=model_config.use_amp,
            vocab_size=model_config.vocab_size,  # Add vocab_size from tokenizer
            use_megatron=True,
            tensor_parallel_size=min(configurator.config.num_gpus, 8),
            pipeline_parallel_size=1
        )
        
        # Create model with Megatron support
        model = create_model(adaptive_config, "moe")
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Train with new pipeline
        model, final_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=adaptive_config,
            device=device
        )
    else:
        # Use legacy training pipeline
        model, final_metrics = train_moe_model(model_config, train_loader, val_loader)
    
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
