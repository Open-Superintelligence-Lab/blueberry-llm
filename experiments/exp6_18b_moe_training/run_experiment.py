import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from experiments.exp6_18b_moe_training.config_18b import MoE18BConfig
from experiments.exp6_18b_moe_training.trainer_18b import train_18b_model
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from utils.helpers import set_seed


def main():
    """Main training script for 18B MoE model"""
    print(f"\n{'='*70}")
    print(f"🚀 18B MoE Training Experiment")
    print(f"{'='*70}\n")
    
    # Check system
    print(f"🔍 System Information:")
    if torch.cuda.is_available():
        print(f"   Device: CUDA")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total Memory: {total_memory:.1f} GB")
        
        if total_memory < 180:
            print(f"\n⚠️  WARNING: This config is optimized for 192GB (B200)")
            print(f"   You have {total_memory:.1f}GB. Consider reducing batch_size or n_layers.")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    else:
        print(f"   Device: CPU")
        print(f"\n❌ ERROR: This model requires a GPU with ~192GB VRAM")
        print(f"   Cannot train 18B model on CPU")
        return

    # Set seed for reproducibility
    set_seed(42)

    # Load data to get vocab_size
    print(f"\n📚 Loading data...")
    temp_config = MoE18BConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Total tokens: {len(tokens):,}")

    # Create config with vocab_size
    config = MoE18BConfig(vocab_size=vocab_size)
    
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
        save_dir=str(checkpoint_dir)
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

