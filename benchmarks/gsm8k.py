"""
GSM8K Benchmark
Evaluates models on grade school math word problems
Requires generation and answer extraction

Usage:
    python benchmarks/gsm8k.py --checkpoint checkpoints/final_model.pt
    python benchmarks/gsm8k.py --checkpoint checkpoints/final_model.pt --max-samples 10
"""

import torch
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from common import load_model_from_checkpoint, get_device_and_dtype


def extract_answer(completion):
    """Extract numerical answer from model completion"""
    # Look for the last number in the text if no #### pattern found
    # Standard GSM8K format uses #### to mark the answer
    match = re.search(r"####\s*(\-?[0-9\.\,]+)", completion)
    if match:
        return match.group(1).replace(',', '')
    
    # Fallback: look for the last number in the text
    # This is a very rough heuristic for 0-shot models that might not follow format
    numbers = re.findall(r"(\-?[0-9\.\,]+)", completion)
    if numbers:
        return numbers[-1].replace(',', '')
        
    return None


def extract_ground_truth(answer_text):
    """Extract ground truth number from GSM8K answer field"""
    match = re.search(r"####\s*(\-?[0-9\.\,]+)", answer_text)
    if match:
        return match.group(1).replace(',', '')
    return None


def generate_solution(model, tokenizer, question, device='cuda', max_new_tokens=256):
    """Generate a solution for a math problem via greedy decoding"""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    
    # Simple greedy generation loop since model doesn't support .generate() natively yet
    curr_ids = input_ids
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Forward pass
                logits = model(curr_ids, return_aux_loss=False)
                
                # Get next token (greedy)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Append
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                
                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
    
    # Decode
    full_text = tokenizer.decode(curr_ids[0], skip_special_tokens=True)
    # Extract just the generated part (remove prompt)
    generated_text = full_text[len(prompt):].strip()
    
    return generated_text


def evaluate_gsm8k(model, tokenizer, split='test', max_samples=None, device='cuda'):
    """Evaluate model on GSM8K dataset"""
    print(f"\nLoading GSM8K dataset ({split} split)...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    results_list = []
    
    for sample in tqdm(dataset, desc="Evaluating GSM8K"):
        question = sample['question']
        gt_text = sample['answer']
        gt_ans = extract_ground_truth(gt_text)
        
        # Generate
        generated_text = generate_solution(model, tokenizer, question, device)
        
        # Extract answer
        pred_ans = extract_answer(generated_text)
        
        # Compare (exact match of string representation of number)
        is_correct = False
        if pred_ans is not None and gt_ans is not None:
            try:
                # Compare as floats to handle 42.0 vs 42
                is_correct = abs(float(pred_ans) - float(gt_ans)) < 1e-6
            except ValueError:
                is_correct = (pred_ans == gt_ans)
        
        if is_correct:
            correct += 1
        total += 1
        
        results_list.append({
            'question': question,
            'ground_truth_full': gt_text,
            'ground_truth': gt_ans,
            'generated_text': generated_text,
            'predicted': pred_ans,
            'is_correct': is_correct
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        'dataset': 'GSM8K',
        'split': split,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_percent': accuracy * 100,
        'samples': results_list,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on GSM8K')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None = all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: auto-generated)')
    args = parser.parse_args()
    
    print("="*70)
    print("GSM8K Benchmark")
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
    
    results = evaluate_gsm8k(
        model, tokenizer,
        split=args.split,
        max_samples=args.max_samples,
        device=device
    )
    
    # Add checkpoint info
    results['checkpoint_path'] = str(args.checkpoint)
    results['model_info'] = {
        'hidden_size': getattr(config, 'hidden_size', 'N/A'),
        'num_layers': getattr(config, 'n_layers', 'N/A') if hasattr(config, 'n_layers') else getattr(config, 'num_hidden_layers', 'N/A'),
        'num_heads': getattr(config, 'n_heads', 'N/A') if hasattr(config, 'n_heads') else getattr(config, 'num_attention_heads', 'N/A'),
    }
    
    # Print results
    print("\n" + "="*70)
    print("GSM8K Results")
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
        output_path = results_dir / f"gsm8k_{args.split}_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print example predictions
    print("\n" + "="*70)
    print("Example Predictions (first 3)")
    print("="*70)
    
    for i, sample in enumerate(results['samples'][:3]):
        print(f"\n[{i+1}] Question: {sample['question'][:80]}...")
        print(f"    Expected: {sample['ground_truth']}")
        print(f"    Predicted: {sample['predicted']}")
        print(f"    {'✓ CORRECT' if sample['is_correct'] else '✗ WRONG'}")
        print(f"    Gen Text: {sample['generated_text'][:100].replace(chr(10), ' ')}...")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
