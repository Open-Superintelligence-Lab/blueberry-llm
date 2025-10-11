"""
Load and use trained Gated DeltaNet model for inference
"""

import torch
import sys
import os
from pathlib import Path

# Fix tokenizer warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp6_gated_deltanet_training.models import GatedDeltaNetWrapper
from experiments.exp6_gated_deltanet_training.config import ExperimentConfig
from transformers import AutoTokenizer


def load_model(checkpoint_path, device='cuda'):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., 'checkpoints/best_model.pt')
        device: Device to load model on
    
    Returns:
        model: Loaded model ready for inference
        config: Model configuration
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint (allowlist ExperimentConfig for PyTorch 2.6+ safety)
    torch.serialization.add_safe_globals([ExperimentConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint['config']
    
    # Create model
    model = GatedDeltaNetWrapper(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Steps trained: {checkpoint['global_step']}")
    print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, device='cuda'):
    """
    Generate text from a prompt
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input text prompt
        max_length: Maximum length to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        device: Device
    
    Returns:
        generated_text: Generated text string
    """
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_length} tokens...")
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we hit max length or EOS
            if input_ids.shape[1] >= 1024:  # Max context length
                break
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """Example usage"""
    print("="*70)
    print("Gated DeltaNet Inference Example")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first by running: python run_experiment.py")
        return
    
    model, config = load_model(checkpoint_path, device)
    
    # Load tokenizer (same as used in training)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Generate text
    print("\n" + "="*70)
    print("Text Generation Examples")
    print("="*70)
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In the year 2050",
    ]
    
    for prompt in prompts:
        print("\n" + "-"*70)
        generated = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=50,
            temperature=0.8,
            top_k=40,
            device=device
        )
        print(f"\nGenerated:\n{generated}")
    
    print("\n" + "="*70)
    print("✅ Inference completed!")
    print("="*70)


if __name__ == "__main__":
    main()

