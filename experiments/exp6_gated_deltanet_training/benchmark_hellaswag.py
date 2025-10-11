"""
Benchmark trained Gated DeltaNet model on HellaSwag dataset
HellaSwag tests commonsense reasoning via sentence completion
"""

import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Fix tokenizer warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp6_gated_deltanet_training.models import GatedDeltaNetWrapper
from experiments.exp6_gated_deltanet_training.config import ExperimentConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
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
    
    # Move to device, convert to bfloat16, and set to eval mode
    model = model.to(device)
    model = model.to(torch.bfloat16)  # DeltaNet requires bfloat16
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Steps trained: {checkpoint['global_step']}")
    print(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Precision: bfloat16")
    
    return model, config


def compute_perplexity(model, tokenizer, text, device='cuda'):
    """
    Compute perplexity of a text sequence
    Lower perplexity = higher likelihood
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = inputs['input_ids'].to(device)
    
    # Skip if too short
    if input_ids.shape[1] < 2:
        return float('inf')
    
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
    
    return perplexity


def evaluate_hellaswag_sample(model, tokenizer, sample, device='cuda'):
    """
    Evaluate a single HellaSwag sample
    Returns: predicted_idx, correct_idx, is_correct
    """
    # Get context and endings
    context = sample['ctx']
    endings = sample['endings']
    correct_idx = int(sample['label'])
    
    # Compute perplexity for each ending
    perplexities = []
    for ending in endings:
        # Full text is context + ending
        full_text = context + " " + ending
        ppl = compute_perplexity(model, tokenizer, full_text, device)
        perplexities.append(ppl)
    
    # Choose ending with lowest perplexity (highest likelihood)
    predicted_idx = np.argmin(perplexities)
    is_correct = (predicted_idx == correct_idx)
    
    return predicted_idx, correct_idx, is_correct, perplexities


def benchmark_hellaswag(model, tokenizer, num_samples=100, device='cuda', split='validation'):
    """
    Benchmark model on HellaSwag dataset
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        num_samples: Number of samples to evaluate (default 100)
        device: Device
        split: Dataset split ('validation' or 'train')
    
    Returns:
        results: Dict with accuracy and details
    """
    print(f"\n{'='*70}")
    print(f"HellaSwag Benchmark - {num_samples} samples")
    print(f"{'='*70}")
    
    # Load HellaSwag dataset
    print(f"\nLoading HellaSwag dataset (split: {split})...")
    dataset = load_dataset("Rowan/hellaswag", split=split)
    print(f"✅ Dataset loaded: {len(dataset)} total samples")
    
    # Limit to num_samples
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    print(f"Evaluating on {len(dataset)} samples...\n")
    
    # Evaluate
    results = []
    correct = 0
    
    model.eval()
    
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        predicted_idx, correct_idx, is_correct, perplexities = evaluate_hellaswag_sample(
            model, tokenizer, sample, device
        )
        
        if is_correct:
            correct += 1
        
        results.append({
            'sample_idx': i,
            'activity': sample['activity_label'],
            'context': sample['ctx'][:100] + "...",  # Truncate for display
            'endings': sample['endings'],
            'predicted_idx': predicted_idx,
            'correct_idx': correct_idx,
            'is_correct': is_correct,
            'perplexities': perplexities,
        })
    
    # Compute accuracy
    accuracy = correct / len(dataset) * 100
    
    # Summary
    summary = {
        'num_samples': len(dataset),
        'num_correct': correct,
        'num_incorrect': len(dataset) - correct,
        'accuracy': accuracy,
        'results': results,
    }
    
    return summary


def print_results(summary):
    """Print benchmark results"""
    print(f"\n{'='*70}")
    print("HellaSwag Benchmark Results")
    print(f"{'='*70}")
    
    print(f"\nOverall Performance:")
    print(f"  Total samples: {summary['num_samples']}")
    print(f"  Correct: {summary['num_correct']}")
    print(f"  Incorrect: {summary['num_incorrect']}")
    print(f"  Accuracy: {summary['accuracy']:.2f}%")
    
    # Random baseline is 25% (4 choices)
    print(f"\n  Random Baseline: 25.00%")
    print(f"  Improvement: {summary['accuracy'] - 25:.2f}%")
    
    # Show some examples
    print(f"\n{'='*70}")
    print("Sample Results (first 5 examples)")
    print(f"{'='*70}")
    
    for i, result in enumerate(summary['results'][:5]):
        print(f"\nExample {i+1}:")
        print(f"  Activity: {result['activity']}")
        print(f"  Context: {result['context']}")
        print(f"  Predicted: {result['predicted_idx']} | Correct: {result['correct_idx']}")
        print(f"  Status: {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}")
        print(f"  Perplexities: {[f'{p:.2f}' for p in result['perplexities']]}")
    
    print(f"\n{'='*70}")


def main():
    """Main benchmarking script"""
    print("="*70)
    print("Gated DeltaNet - HellaSwag Benchmark")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Checkpoint path - use latest/best checkpoint
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    
    # Try final_model.pt first, then best_model.pt
    checkpoint_path = checkpoint_dir / "final_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first by running: python run_experiment.py")
        return
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Load tokenizer (same as used in training)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Run benchmark
    summary = benchmark_hellaswag(
        model=model,
        tokenizer=tokenizer,
        num_samples=100,  # Evaluate on 100 samples
        device=device,
        split='validation'  # Use validation split
    )
    
    # Print results
    print_results(summary)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "hellaswag_benchmark.txt"
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HellaSwag Benchmark Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {checkpoint_path.name}\n")
        f.write(f"Samples: {summary['num_samples']}\n")
        f.write(f"Correct: {summary['num_correct']}\n")
        f.write(f"Accuracy: {summary['accuracy']:.2f}%\n")
        f.write(f"Random Baseline: 25.00%\n")
        f.write(f"Improvement: {summary['accuracy'] - 25:.2f}%\n")
    
    print(f"\n✅ Results saved to: {results_file}")
    print("="*70)


if __name__ == "__main__":
    main()

