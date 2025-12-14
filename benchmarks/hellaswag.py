"""
HellaSwag Benchmark
Evaluates models on commonsense reasoning via sentence completion

Usage:
    python benchmarks/hellaswag.py --checkpoint experiments/exp7_*/checkpoints/best_model.pt
    python benchmarks/hellaswag.py --checkpoint exp7/checkpoints/best_model.pt --split validation
    python benchmarks/hellaswag.py --checkpoint exp7/checkpoints/best_model.pt --max-samples 100
"""

import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from common import load_model_from_checkpoint, get_device_and_dtype


def compute_perplexity(model, tokenizer, text, device='cuda'):
    """
    Compute perplexity of a text sequence
    Lower perplexity = higher likelihood
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = inputs['input_ids'].to(device)
    
    if input_ids.shape[1] < 2:
        return float('inf')
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Model does not accept labels or return loss object, so we compute it manually
            logits = model(input_ids, return_aux_loss=False)
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            perplexity = torch.exp(loss).item()
    
    return perplexity


def evaluate_hellaswag_sample(model, tokenizer, sample, device='cuda'):
    """Evaluate a single HellaSwag sample"""
    context = sample['ctx']
    endings = sample['endings']
    correct_idx = int(sample['label'])
    
    perplexities = []
    for ending in endings:
        full_text = context + " " + ending
        ppl = compute_perplexity(model, tokenizer, full_text, device)
        perplexities.append(ppl)
    
    predicted_idx = np.argmin(perplexities)
    is_correct = (predicted_idx == correct_idx)
    
    return predicted_idx, correct_idx, is_correct, perplexities


def evaluate_hellaswag(model, tokenizer, split='validation', max_samples=None, device='cuda'):
    """Evaluate model on HellaSwag dataset"""
    print(f"\nLoading HellaSwag dataset ({split} split)...")
    dataset = load_dataset("Rowan/hellaswag", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    results_list = []
    
    for sample in tqdm(dataset, desc="Evaluating HellaSwag"):
        predicted_idx, correct_idx, is_correct, perplexities = evaluate_hellaswag_sample(
            model, tokenizer, sample, device
        )
        
        if is_correct:
            correct += 1
        total += 1
        
        results_list.append({
            'activity': sample['activity_label'],
            'context': sample['ctx'],
            'endings': sample['endings'],
            'predicted_idx': int(predicted_idx),
            'correct_idx': correct_idx,
            'is_correct': bool(is_correct),
            'perplexities': [float(p) for p in perplexities],
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        'dataset': 'HellaSwag',
        'split': split,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_percent': accuracy * 100,
        'samples': results_list,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on HellaSwag')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='validation',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None = all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: auto-generated)')
    args = parser.parse_args()
    
    print("="*70)
    print("HellaSwag Benchmark")
    print("="*70)
    
    # Setup device and dtype
    device, dtype = get_device_and_dtype()
    print(f"Device: {device}")
    
    # Load model
    model, config, tokenizer = load_model_from_checkpoint(
        args.checkpoint, device=device, dtype=dtype
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("Running Evaluation")
    print("="*70)
    
    results = evaluate_hellaswag(
        model, tokenizer,
        split=args.split,
        max_samples=args.max_samples,
        device=device
    )
    
    # Add checkpoint info to results
    results['checkpoint_path'] = str(args.checkpoint)
    results['model_info'] = {
        'hidden_size': getattr(config, 'hidden_size', 'N/A'),
        'num_layers': getattr(config, 'num_hidden_layers', 'N/A'),
        'num_heads': getattr(config, 'num_attention_heads', 'N/A'),
    }
    
    # Print results
    print("\n" + "="*70)
    print("HellaSwag Results")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {results['split']}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy_percent']:.2f}%")
    print("="*70)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_path = Path(args.checkpoint)
        exp_dir = checkpoint_path.parent.parent
        results_dir = exp_dir / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        output_path = results_dir / f"hellaswag_{args.split}_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print example predictions
    print("\n" + "="*70)
    print("Example Predictions (first 3)")
    print("="*70)
    
    for i, sample in enumerate(results['samples'][:3]):
        print(f"\n[{i+1}] Activity: {sample['activity']}")
        print(f"    Context: {sample['context'][:80]}...")
        print(f"    Endings: ")
        for j, ending in enumerate(sample['endings']):
            marker = "→" if j == sample['predicted_idx'] else " "
            correct_marker = "*" if j == sample['correct_idx'] else " "
            print(f"      {marker} [{j}] {correct_marker} {ending[:60]}...")
        print(f"    {'✓ CORRECT' if sample['is_correct'] else '✗ WRONG'}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

