"""
Inference script for Reasoning Architecture
Load trained model and generate text samples
"""

import torch
import sys
import os
from pathlib import Path

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp8_reasoning_architecture.config import ExperimentConfig
from experiments.exp8_reasoning_architecture.models import ReasoningModelWrapper
from transformers import AutoTokenizer


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    torch.serialization.add_safe_globals([ExperimentConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    
    # Create model
    model = ReasoningModelWrapper(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Step: {checkpoint.get('global_step', 'unknown')}")
    print(f"  Val Loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    return model, config


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, device='cuda'):
    """Generate text from a prompt"""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_length} tokens...\n")
    print("="*70)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    print(generated_text)
    print("="*70)
    
    return generated_text


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate text with Reasoning Model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Text prompt for generation')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try relative to script directory
        checkpoint_path = Path(__file__).parent / args.checkpoint
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return
    
    model, config = load_model(checkpoint_path, device)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Print model info
    model.print_info()
    
    # Generate
    generate_text(
        model, 
        tokenizer, 
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )


if __name__ == '__main__':
    main()

