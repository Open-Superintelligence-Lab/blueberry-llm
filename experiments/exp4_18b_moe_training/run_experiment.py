import torch
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from experiments.exp4_18b_moe_training.config_18b import MoE18BConfig
from experiments.exp4_18b_moe_training.config_4090 import MoE4090Config
from experiments.exp4_18b_moe_training.trainer_18b import train_18b_model
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from utils.helpers import set_seed


def main():
    """Main training script for MoE model with configurable GPU support"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MoE model on different GPUs')
    parser.add_argument('--gpu', type=str, default='b200', choices=['b200', '4090'],
                      help='GPU configuration to use (b200 or 4090)')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from (e.g., checkpoints/checkpoint_latest.pt)')
    args = parser.parse_args()
    
    # Select config based on GPU type
    if args.gpu == '4090':
        ConfigClass = MoE4090Config
        model_name = "MoE 4090"
        expected_memory = 24
    else:
        ConfigClass = MoE18BConfig
        model_name = "MoE 18B (B200)"
        expected_memory = 192
    
    print(f"\n{'='*70}")
    print(f"🚀 {model_name} Training Experiment")
    print(f"{'='*70}\n")
    
    # Check system
    print(f"🔍 System Information:")
    if torch.cuda.is_available():
        print(f"   Device: CUDA")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total Memory: {total_memory:.1f} GB")
        
        # Adjusted memory check based on selected config
        memory_threshold = expected_memory * 0.9  # 90% of expected
        if total_memory < memory_threshold:
            print(f"\n⚠️  WARNING: This config is optimized for {expected_memory}GB ({args.gpu.upper()})")
            print(f"   You have {total_memory:.1f}GB. Training may fail due to OOM.")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    else:
        print(f"   Device: CPU")
        print(f"\n❌ ERROR: This model requires a GPU")
        print(f"   Cannot train MoE model on CPU")
        return

    # Set seed for reproducibility
    set_seed(42)

    # Load data to get vocab_size
    print(f"\n📚 Loading data...")
    temp_config = ConfigClass()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Total tokens: {len(tokens):,}")

    # Create config with vocab_size
    config = ConfigClass(vocab_size=vocab_size)
    
    # Create dataset
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    # Train/val split (90/10)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples: {len(val_dataset):,}")

    # Create data loaders
    # Use num_workers=0 for very large models to avoid multiprocessing memory overhead
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    # Create checkpoint directory
    exp_dir = Path(__file__).parent
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\n💾 Checkpoints will be saved to: {checkpoint_dir}")

    # Train model
    print(f"\n🎯 Starting training...")
    model, final_metrics = train_18b_model(
        config, 
        train_loader, 
        val_loader, 
        save_dir=str(checkpoint_dir),
        checkpoint_path=args.resume
    )

    print(f"\n{'='*70}")
    print(f"✅ Experiment Complete!")
    print(f"{'='*70}")
    print(f"\nFinal Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    print(f"\nCheckpoints saved in: {checkpoint_dir}")
    print(f"Results saved in: {checkpoint_dir}/training_results.json")


if __name__ == "__main__":
    main()

