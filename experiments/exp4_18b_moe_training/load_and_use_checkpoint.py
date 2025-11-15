"""
Example script showing how to load a trained checkpoint for inference or continued training
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp4_18b_moe_training.models_18b import MoE18BLLM
from experiments.exp4_18b_moe_training.trainer_18b import load_checkpoint


def load_model_for_inference(checkpoint_path, device='cuda'):
    """
    Load a trained model from checkpoint for inference (no optimizer/scheduler needed)
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., 'checkpoints/checkpoint_latest.pt')
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model for inference...")
    
    # First load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"\n📋 Model Configuration:")
    print(f"   Parameters: {config.total_params/1e9:.2f}B total, {config.active_params/1e9:.2f}B active")
    print(f"   Architecture: {config.n_layers} layers, {config.d_model} hidden, {config.num_experts} experts")
    
    # Create model
    model = MoE18BLLM(config)
    
    # Load checkpoint (weights only, no optimizers)
    step, _, metrics = load_checkpoint(checkpoint_path, model, device=device)
    
    # Set to eval mode
    model.eval()
    
    print(f"✅ Model loaded and ready for inference!")
    return model, config, step


def generate_text(model, config, tokenizer, prompt, max_tokens=100, temperature=1.0, device='cuda'):
    """
    Generate text from the model
    
    Args:
        model: Trained model
        config: Model config
        tokenizer: Tokenizer
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n🎯 Generating from prompt: '{prompt}'")
    print(f"Generating {max_tokens} tokens...")
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits
            logits, _ = model(tokens[:, -config.max_seq_len:], return_aux_loss=False)
            
            # Sample next token
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Check for end of sequence (if you have an EOS token)
            # if next_token.item() == tokenizer.eos_token_id:
            #     break
    
    # Decode
    generated_tokens = tokens[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def example_resume_training():
    """
    Example: Resume training from a checkpoint
    """
    from torch.utils.data import DataLoader
    from data.loader import load_and_cache_data
    from data.dataset import TextTokenDataset
    from experiments.exp4_18b_moe_training.config_4090 import MoE4090Config
    from experiments.exp4_18b_moe_training.trainer_18b import train_18b_model
    
    print("\n" + "="*70)
    print("Example: Resuming Training from Checkpoint")
    print("="*70 + "\n")
    
    # Specify checkpoint path
    checkpoint_path = "experiments/exp4_18b_moe_training/checkpoints/checkpoint_latest.pt"
    
    # Load config (you can also load from checkpoint)
    config = MoE4090Config(vocab_size=49152)  # Set vocab_size from your data
    
    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    # Split data
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Resume training from checkpoint
    model, final_metrics = train_18b_model(
        config, 
        train_loader, 
        val_loader,
        save_dir='experiments/exp4_18b_moe_training/checkpoints',
        checkpoint_path=checkpoint_path  # This resumes from checkpoint!
    )
    
    print("✅ Training resumed and completed!")


def example_inference():
    """
    Example: Load model and run inference
    """
    print("\n" + "="*70)
    print("Example: Loading Model for Inference")
    print("="*70 + "\n")
    
    # Path to checkpoint
    checkpoint_path = "experiments/exp4_18b_moe_training/checkpoints/checkpoint_latest.pt"
    
    # Load model
    model, config, step = load_model_for_inference(checkpoint_path)
    
    print(f"\nModel was trained for {step} steps")
    print(f"Model has {config.total_params/1e9:.2f}B parameters")
    
    # To use for inference, you would do:
    # generated_text = generate_text(model, config, tokenizer, "Your prompt here")
    
    print("\n💡 Tip: Use this model with generate_text() function for inference")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and use trained checkpoint')
    parser.add_argument('--mode', type=str, default='inference', 
                       choices=['inference', 'resume'],
                       help='Mode: inference or resume training')
    parser.add_argument('--checkpoint', type=str, 
                       default='experiments/exp4_18b_moe_training/checkpoints/checkpoint_latest.pt',
                       help='Path to checkpoint file')
    
    args = parser.parse_args()
    
    if args.mode == 'inference':
        example_inference()
    else:
        example_resume_training()

