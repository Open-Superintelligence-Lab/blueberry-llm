import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp4_18b_moe_training.models_18b import MoE18BLLM
from transformers import AutoTokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    """Generate text from the model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\n{'='*70}")
    print(f"🎨 Generating text...")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"\nGenerated text:")
    print(f"{prompt}", end='', flush=True)
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model predictions
            logits, _ = model(input_ids, return_aux_loss=False)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Decode and print
            next_word = tokenizer.decode(next_token.item())
            print(next_word, end='', flush=True)
            
            # Stop if we generate EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print(f"\n{'='*70}\n")
    
    return tokenizer.decode(input_ids[0])


def run_inference_demo(model, tokenizer, config):
    """Run inference demonstrations"""
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a galaxy far, far away",
        "def fibonacci(n):",
    ]
    
    print(f"\n{'='*70}")
    print(f"🚀 Running Inference Demonstrations")
    print(f"{'='*70}")
    print(f"\nModel: {config.total_params/1e9:.2f}B parameters")
    print(f"Active per forward: {config.active_params/1e9:.2f}B parameters")
    print(f"Using {config.num_experts} experts, top-{config.expert_top_k} routing")
    
    for prompt in prompts:
        generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.9, top_k=50)
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on trained MoE model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_latest.pt',
                      help='Path to checkpoint file')
    parser.add_argument('--prompt', type=str, default=None,
                      help='Custom prompt for generation')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🤖 MoE Model Inference")
    print(f"{'='*70}\n")
    
    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint
    checkpoint_path = Path(__file__).parent / args.checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    config = checkpoint['config']
    
    # Create model
    print(f"🚀 Initializing model...")
    model = MoE18BLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    
    if args.prompt:
        # Generate from custom prompt
        generate_text(model, tokenizer, args.prompt, max_new_tokens=100, temperature=0.8)
    else:
        # Run demo with multiple prompts
        run_inference_demo(model, tokenizer, config)

