#!/usr/bin/env python3
"""
Inference script for Blueberry LLM
Usage: python3 inference.py "Your prompt here"
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from legacy.llm import MoEMinimalLLM, MoEModelConfig
from core.auto_config import AutoConfig
from transformers import AutoTokenizer
import argparse

def load_model(checkpoint_path="blueberry_model.pt"):
    """Load the trained model"""
    print(f"📦 Loading model from {checkpoint_path}...")
    
    # Add safe globals for custom classes
    from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
    torch.serialization.add_safe_globals([MoEModelConfig, AutoConfig, GPT2TokenizerFast])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Handle tokenizer - create if not in checkpoint
    if 'tokenizer' in checkpoint:
        tokenizer = checkpoint['tokenizer']
        print("✅ Loaded tokenizer from checkpoint")
    else:
        print("⚠️  Tokenizer not found in checkpoint, creating new one...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Created tokenizer from HuggingFaceTB/SmolLM-135M")
    
    # Create model
    model = MoEMinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"   Model: {config.d_model}d × {config.n_layers}L × {config.n_heads}H")
    print(f"   Experts: {config.num_experts}")
    print(f"   Vocab size: {config.vocab_size}")
    
    return model, tokenizer, device, config

def generate_text(model, tokenizer, device, prompt, max_length=100, temperature=0.8, top_k=50):
    """Generate text from a prompt"""
    
    # Tokenize prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\n🎯 Generating from prompt: '{prompt}'")
    print(f"📝 Max length: {max_length}, Temperature: {temperature}, Top-k: {top_k}")
    print("-" * 50)
    
    generated = inputs.clone()
    
    with torch.no_grad():
        for i in range(max_length):
            # Forward pass
            logits = model(generated, return_aux_loss=False)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Print token as we generate
            token_text = tokenizer.decode(next_token.item())
            print(token_text, end='', flush=True)
            
            # Stop if we hit end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print("\n" + "-" * 50)
    
    # Decode full sequence
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt):]
    
    return generated_text

def interactive_mode(model, tokenizer, device, config):
    """Interactive chat mode"""
    print("\n🤖 Interactive mode - type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n👤 You: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt.strip():
                continue
                
            print("🤖 Bot: ", end='')
            generated = generate_text(
                model, tokenizer, device, prompt, 
                max_length=50, temperature=0.8, top_k=50
            )
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inference for Blueberry LLM")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--checkpoint", default="blueberry_model.pt", help="Model checkpoint path")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    
    args = parser.parse_args()
    
    # Load model
    try:
        model, tokenizer, device, config = load_model(args.checkpoint)
    except FileNotFoundError:
        print(f"❌ Model file '{args.checkpoint}' not found!")
        print("💡 Make sure you've trained a model first with: python3 train_auto.py")
        return
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, device, config)
    elif args.prompt:
        # Single prompt
        generated = generate_text(
            model, tokenizer, device, args.prompt,
            max_length=args.max_length, 
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"\n📄 Full generated text:\n{generated}")
    else:
        # Default examples
        examples = [
            "The future of artificial intelligence is",
            "In a world where technology",
            "The most important thing in life is",
        ]
        
        for prompt in examples:
            generated = generate_text(
                model, tokenizer, device, prompt,
                max_length=50, temperature=0.8, top_k=50
            )
            print()

if __name__ == "__main__":
    main()
