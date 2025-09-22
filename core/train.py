#!/usr/bin/env python3
"""
Main training script for blueberry-llm.

This script provides a clean entry point for training GPU-adaptive LLM models
with automatic optimization based on hardware capabilities.

Usage:
    python train.py                    # Use default MoE configuration
    python train.py --config dev       # Use development configuration
    python train.py --config rtx5090   # Use RTX 5090 optimized configuration
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import time

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

# Import our modular components
from configs import AdaptiveMoEModelConfig, get_rtx4090_config, get_rtx5090_config, get_development_config
from data import load_and_cache_data, TextTokenDataset
from models import AdaptiveMoEMinimalLLM, create_model
from training import train_model, validate_training_setup
from system import print_system_info, SYSTEM_CONFIG


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GPU-adaptive LLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        choices=["default", "dev", "rtx4090", "rtx5090"],
        help="Configuration preset to use"
    )
    
    # Model options
    parser.add_argument("--model-type", type=str, default="moe", choices=["moe", "standard"],
                       help="Type of model to train")
    parser.add_argument("--d-model", type=int, help="Model dimension")
    parser.add_argument("--n-layers", type=int, help="Number of layers")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads")
    parser.add_argument("--num-experts", type=int, help="Number of experts (MoE only)")
    
    # Training options
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 acceleration")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    
    # Data options
    parser.add_argument("--num-documents", type=int, help="Number of documents to load")
    parser.add_argument("--max-tokens", type=int, help="Maximum number of tokens")
    parser.add_argument("--max-seq-len", type=int, help="Maximum sequence length")
    
    # System options
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate-setup", action="store_true", help="Validate setup before training")
    parser.add_argument("--no-megatron", action="store_true", help="Force native backend (disable Megatron)")
    parser.add_argument("--use-megatron", action="store_true", help="Force Megatron backend (enable Megatron)")
    
    return parser.parse_args()


def get_config(args):
    """Get configuration based on arguments."""
    # Base configuration
    if args.config == "dev":
        config = get_development_config()
    elif args.config == "rtx4090":
        config = get_rtx4090_config()
    elif args.config == "rtx5090":
        config = get_rtx5090_config()
    else:
        config = AdaptiveMoEModelConfig()
    
    # Override with command line arguments
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.n_heads is not None:
        config.n_heads = args.n_heads
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.muon_lr = args.lr
    if args.num_documents is not None:
        config.num_documents = args.num_documents
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len
    
    # Disable features if requested
    if args.no_fp8:
        config.use_fp8 = False
    if args.no_amp:
        config.use_amp = False
    if args.no_megatron:
        config.use_megatron = False
    if args.use_megatron:
        config.use_megatron = True
    
    # Re-run post_init to update dependent values
    config.__post_init__()
    
    return config


def setup_data(config: AdaptiveMoEModelConfig):
    """Setup data loaders."""
    print("\n📚 Setting up data...")
    
    # Load and cache data
    texts, tokenizer, tokens = load_and_cache_data(config)
    
    # Create dataset
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    # Train/validation split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✅ Dataset: {len(train_dataset):,} train, {len(val_dataset):,} val samples")
    print(f"✅ Vocab size: {config.vocab_size:,}")
    
    return train_loader, val_loader, tokenizer


def setup_model(config: AdaptiveMoEModelConfig, model_type: str):
    """Setup model."""
    print(f"\n🤖 Setting up {model_type} model...")
    
    # Create model
    model = create_model(config, model_type)
    
    # Move to appropriate device
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ Model moved to GPU")
    else:
        print(f"⚠️ Using CPU (CUDA not available)")
    
    return model


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Print header
    print("🚀 Blueberry LLM Training")
    print("=" * 60)
    
    # Print system information
    print_system_info()
    
    # Get configuration
    config = get_config(args)
    
    # Setup data
    train_loader, val_loader, tokenizer = setup_data(config)
    
    # Setup model
    model = setup_model(config, args.model_type)
    
    # Validate setup if requested
    if args.validate_setup:
        print("\n🔍 Validating training setup...")
        if not validate_training_setup(model, train_loader, val_loader, config):
            print("❌ Setup validation failed. Exiting.")
            return
        print("✅ Setup validation successful!")
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🎯 Starting training on {device}")
    print(f"📋 Configuration: {args.config}")
    print(f"🤖 Model type: {args.model_type}")
    
    # Start training
    start_time = time.time()
    
    try:
        model, final_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        training_time = time.time() - start_time
        
        # Print final results
        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")
        print(f"⏱️ Total training time: {training_time/60:.1f} minutes")
        print("\n📊 Final Results:")
        print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
        
        if 'mfu' in final_metrics:
            print(f"   Model FLOPs Utilization: {final_metrics['mfu']*100:.1f}%")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
