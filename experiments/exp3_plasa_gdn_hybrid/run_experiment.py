"""
Comprehensive Experiment: Test Adaptive Sparse Attention vs Full Attention
- 4 Original: full_attention + linear_attention
- 4 PLASA: plasa_attention (Per-Layer Adaptive Sparse) + linear_attention

PLASA uses PROGRESSIVE_SPARSE schedule:
- Early layers: Dense (k=L)
- Middle layers: Aggressive sparse (k=L/4)
- Late layers: Moderate sparse (k=L/2)

Total: 8 architecture variants
"""

import torch
import sys
import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp3_plasa_gdn_hybrid.models import EnhancedQwen3NextForCausalLM
from models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from models.qwen3_next.modular_qwen3_next import Qwen3NextForCausalLM
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from utils.helpers import set_seed
from torch.utils.data import DataLoader


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, dataloader, device, max_batches=None):
    """Evaluate the model and return metrics"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            # Handle tuple output from dataset (x, y)
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            # Shift for next-token prediction
            shift_preds = predictions[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            correct = (shift_preds == shift_labels).sum().item()
            
            total_loss += loss.item() * input_ids.numel()
            total_correct += correct
            total_tokens += shift_labels.numel()
            
            # Clean up to prevent memory buildup
            del outputs, logits, predictions, shift_preds, shift_labels, input_ids, labels, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
    }


def train_epoch_with_history(model, dataloader, optimizer, device, max_steps, total_steps):
    """Train for one epoch and track loss history per step"""
    model.train()
    total_loss = 0
    total_tokens = 0
    loss_history = []
    
    for i, batch in enumerate(dataloader):
        # Stop if we've reached max_steps across all epochs
        if total_steps >= max_steps:
            break
        
        # Handle tuple output from dataset (x, y)
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        else:
            input_ids = batch.to(device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        total_steps += 1
        
        # Record loss at this step
        step_loss = total_loss / total_tokens
        loss_history.append({'step': total_steps, 'loss': step_loss})
        
        if total_steps % 50 == 0:  # Log every 50 steps for longer training
            print(f"  Step {total_steps}/{max_steps}, Loss: {step_loss:.4f}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, total_steps, loss_history


# 8 Attention Patterns to test
ALL_PATTERNS = {
    # Original patterns (F = full_attention, L = linear_attention)
    "1_all_plasa": ["plasa_attention", "plasa_attention", "plasa_attention", "plasa_attention"],  # Extra pattern with all PLASA
    "2_all_full": ["full_attention", "full_attention", "full_attention", "full_attention"],  # Extra pattern with all Full
    "3_all_linear": ["linear_attention", "linear_attention", "linear_attention", "linear_attention"],  # Extra pattern with all Linear

    # PLASA patterns (P = plasa_attention with PROGRESSIVE_SPARSE, L = linear_attention)
    "4_plasa_sandwich": ["linear_attention", "plasa_attention", "plasa_attention", "linear_attention"],
    "5_plasa_alternating": ["plasa_attention", "linear_attention", "plasa_attention", "linear_attention"],
    "6_plasa_linear_first": ["linear_attention", "linear_attention", "plasa_attention", "plasa_attention"],
    "7_plasa_full_first": ["plasa_attention", "plasa_attention", "linear_attention", "linear_attention"],

    "8_sandwich": ["linear_attention", "full_attention", "full_attention", "linear_attention"],
    "9_alternating": ["full_attention", "linear_attention", "full_attention", "linear_attention"],
    "10_linear_first": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
    "11_full_first": ["full_attention", "full_attention", "linear_attention", "linear_attention"],

}


def test_pattern(pattern_name, layer_types, use_plasa=False):
    """Test a specific attention pattern"""
    print(f"\n{'='*70}")
    print(f"Pattern {pattern_name.upper()}")
    print(f"Layers: {' -> '.join([t[0].upper() for t in layer_types])}")
    print(f"Details: {layer_types}")
    print(f"{'='*70}\n")
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = Qwen3NextConfig(
        vocab_size=50257,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        rope_theta=10000.0,
        attention_dropout=0.1,
        hidden_dropout_prob=0.1,
        rms_norm_eps=1e-6,
        head_dim=32,
        partial_rotary_factor=1.0,
        layer_types=layer_types,
        # Linear attention config
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        # MoE configuration
        num_experts=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_jitter_noise=0.0,
        decoder_sparse_step=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=0,
        mlp_only_layers=[],
        # Adaptive Sparse Attention config
        indexer_heads=4,
        indexer_dim=32,
        sparse_top_k=64,  # Note: Overridden by PROGRESSIVE_SPARSE schedule
    )
    
    # Set attention implementation (required for full_attention layers)
    config._attn_implementation = "eager"
    
    # Load cached data
    from dataclasses import dataclass
    @dataclass
    class SimpleConfig:
        num_documents: int = 1000
        max_tokens: int = 2_000_000
        vocab_size: int = 50257
    
    data_config = SimpleConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    config.vocab_size = len(tokenizer)
    
    max_seq_len = 128
    dataset = TextTokenDataset(tokens, max_seq_len)
    
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Create model - use Enhanced model for all patterns (supports all layer types)
    model = EnhancedQwen3NextForCausalLM(config).to(device)
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Verify layer types
    print("Layer verification: ", end="")
    for i, layer in enumerate(model.model.layers):
        has_self_attn = hasattr(layer, 'self_attn')
        has_linear_attn = hasattr(layer, 'linear_attn')
        has_plasa_attn = hasattr(layer, 'plasa_attn')

        if has_self_attn:
            actual = "F"
        elif has_linear_attn:
            actual = "L"
        elif has_plasa_attn:
            actual = "P"
        else:
            actual = "?"
        print(actual, end=" ")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Train for 1000 steps
    max_steps = 1000
    start_time = time.time()
    total_steps = 0
    
    train_loss, total_steps, loss_history = train_epoch_with_history(model, train_loader, optimizer, device, max_steps, total_steps)
    val_metrics = evaluate(model, val_loader, device, max_batches=100)
    
    training_time = time.time() - start_time
    
    result = {
        'pattern': pattern_name,
        'layer_types': layer_types,
        'uses_plasa': use_plasa,
        'num_parameters': num_params,
        'training_time': training_time,
        'train_loss': train_loss,
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
        'val_perplexity': val_metrics['perplexity'],
        'loss_history': loss_history,
    }
    
    print(f"\n{pattern_name.upper()} Results:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Val Acc: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"  Val PPL: {val_metrics['perplexity']:.2f}")
    print(f"  Time: {training_time:.1f}s")
    
    # Clear GPU memory more thoroughly
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return result


def main():
    print("="*70)
    print("ADAPTIVE SPARSE ATTENTION EXPERIMENT (EXP3)")
    print("="*70)
    print("\nTesting 11 patterns:")
    print("  - 4 Original: Full Attention (F) + Linear Attention (L)")
    print("  - 4 PLASA: Per-Layer Adaptive Sparse (P) + Linear Attention (L)")
    print("  - 3 Extra: All PLASA (P), All Full (F), All Linear (L)")
    print("\nPLASA uses PROGRESSIVE_SPARSE schedule:")
    print("  - Early layers: Dense (k=L)")
    print("  - Middle layers: Aggressive sparse (k=L/4)")
    print("  - Late layers: Moderate sparse (k=L/2)")
    print("\nConfiguration:")
    print("  - 4 layers, 128 hidden dim, 14M parameters")
    print("  - 4 experts, top-2 routing, MoE every 2 layers")
    print("  - 1000 training steps")
    print()

    results = {}
    for pattern_name, layer_types in ALL_PATTERNS.items():
        # Check if this pattern uses PLASA
        use_plasa = "plasa" in pattern_name or any("plasa" in lt for lt in layer_types)
        
        try:
            result = test_pattern(pattern_name, layer_types, use_plasa)
            results[pattern_name] = result
        except Exception as e:
            print(f"\n[ERROR] testing {pattern_name}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
    
    # Summary Tables
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ALL PATTERNS")
    print(f"{'='*70}\n")
    
    # Sort by validation loss (best first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['val_loss'])
    
    print(f"{'Rank':<6} {'Pattern':<20} {'Type':<8} {'Val Loss':<10} {'Val Acc':<10} {'Val PPL':<12} {'Time':<8}")
    print("-" * 90)
    
    for rank, (pattern_name, result) in enumerate(sorted_results, 1):
        pattern_type = "PLASA" if result['uses_plasa'] else "Original"
        medal = "#1" if rank == 1 else "#2" if rank == 2 else "#3" if rank == 3 else f"#{rank}"

        print(f"{medal:<6} {pattern_name:<20} {pattern_type:<8} "
              f"{result['val_loss']:<10.4f} {result['val_accuracy']:<10.4f} "
              f"{result['val_perplexity']:<12.2f} {result['training_time']:<8.1f}s")

    # Category comparison
    print(f"\n{'='*70}")
    print("CATEGORY AVERAGES")
    print(f"{'='*70}\n")

    original_results = [r for r in results.values() if not r['uses_plasa']]
    plasa_results = [r for r in results.values() if r['uses_plasa']]
    
    if original_results:
        avg_orig_loss = sum(r['val_loss'] for r in original_results) / len(original_results)
        avg_orig_acc = sum(r['val_accuracy'] for r in original_results) / len(original_results)
        avg_orig_ppl = sum(r['val_perplexity'] for r in original_results) / len(original_results)
        avg_orig_time = sum(r['training_time'] for r in original_results) / len(original_results)
        
        print(f"Original (Full + Linear Attention):")
        print(f"  Avg Val Loss: {avg_orig_loss:.4f}")
        print(f"  Avg Val Acc:  {avg_orig_acc:.4f} ({avg_orig_acc*100:.2f}%)")
        print(f"  Avg Val PPL:  {avg_orig_ppl:.2f}")
        print(f"  Avg Time:     {avg_orig_time:.1f}s")
    
    if plasa_results:
        avg_plasa_loss = sum(r['val_loss'] for r in plasa_results) / len(plasa_results)
        avg_plasa_acc = sum(r['val_accuracy'] for r in plasa_results) / len(plasa_results)
        avg_plasa_ppl = sum(r['val_perplexity'] for r in plasa_results) / len(plasa_results)
        avg_plasa_time = sum(r['training_time'] for r in plasa_results) / len(plasa_results)

        print(f"\nPLASA (Adaptive Sparse + Linear Attention):")
        print(f"  Avg Val Loss: {avg_plasa_loss:.4f}")
        print(f"  Avg Val Acc:  {avg_plasa_acc:.4f} ({avg_plasa_acc*100:.2f}%)")
        print(f"  Avg Val PPL:  {avg_plasa_ppl:.2f}")
        print(f"  Avg Time:     {avg_plasa_time:.1f}s")

        if original_results:
            print(f"\nPLASA vs Original:")
            print(f"  Loss diff:  {avg_plasa_loss - avg_orig_loss:+.4f}")
            print(f"  Acc diff:   {(avg_plasa_acc - avg_orig_acc)*100:+.2f}%")
            print(f"  PPL diff:   {avg_plasa_ppl - avg_orig_ppl:+.2f}")
            print(f"  Time diff:  {avg_plasa_time - avg_orig_time:+.1f}s")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    with open(results_dir / 'comprehensive_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create loss comparison plot
    print(f"\n{'='*70}")
    print("Creating loss comparison plot...")
    print(f"{'='*70}\n")
    
    plt.figure(figsize=(14, 8))

    # Define colors for different patterns
    original_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue tones
    plasa_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']  # Purple/pink tones

    color_idx_orig = 0
    color_idx_plasa = 0

    for pattern_name, result in results.items():
        if 'loss_history' in result and result['loss_history']:
            steps = [h['step'] for h in result['loss_history']]
            losses = [h['loss'] for h in result['loss_history']]

            # Choose color based on pattern type
            if result['uses_plasa']:
                color = plasa_colors[color_idx_plasa % len(plasa_colors)]
                linestyle = '--'
                color_idx_plasa += 1
            else:
                color = original_colors[color_idx_orig % len(original_colors)]
                linestyle = '-'
                color_idx_orig += 1

            # Create readable pattern string
            layer_types = result['layer_types']
            pattern_str = ' -> '.join([
                'L' if 'linear' in lt else ('P' if 'plasa' in lt else 'F')
                for lt in layer_types
            ])

            label = f"{pattern_name}: {pattern_str}"
            plt.plot(steps, losses, label=label, color=color, linestyle=linestyle, linewidth=2, marker='o', markersize=4)

    plt.xlabel('Training Step', fontsize=12, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Comparison - All 8 Attention Patterns (Exp3)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)

    # Add legend explanation
    textstr = 'Pattern Key:\nF = Full Attention\nL = Linear Attention\nP = PLASA (Adaptive Sparse)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = results_dir / 'loss_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss comparison plot saved to: {plot_path}")
    
    # Also create a separate plot comparing Original vs PLASA averages
    plt.figure(figsize=(14, 8))

    # Calculate average loss curves for each category
    original_losses = {}
    plasa_losses = {}

    for pattern_name, result in results.items():
        if 'loss_history' in result and result['loss_history']:
            for h in result['loss_history']:
                step = h['step']
                loss = h['loss']

                if result['uses_plasa']:
                    if step not in plasa_losses:
                        plasa_losses[step] = []
                    plasa_losses[step].append(loss)
                else:
                    if step not in original_losses:
                        original_losses[step] = []
                    original_losses[step].append(loss)

    # Plot averages
    if original_losses:
        steps = sorted(original_losses.keys())
        avg_losses = [sum(original_losses[s]) / len(original_losses[s]) for s in steps]
        plt.plot(steps, avg_losses, label='Original (Full + Linear Attention)',
                color='#1f77b4', linestyle='-', linewidth=3, marker='o', markersize=6)

    if plasa_losses:
        steps = sorted(plasa_losses.keys())
        avg_losses = [sum(plasa_losses[s]) / len(plasa_losses[s]) for s in steps]
        plt.plot(steps, avg_losses, label='PLASA (Adaptive Sparse + Linear Attention)',
                color='#9467bd', linestyle='--', linewidth=3, marker='s', markersize=6)

    plt.xlabel('Training Step', fontsize=12, fontweight='bold')
    plt.ylabel('Average Training Loss', fontsize=12, fontweight='bold')
    plt.title('Average Training Loss: Original vs PLASA (Exp3)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path_avg = results_dir / 'loss_comparison_average.png'
    plt.savefig(plot_path_avg, dpi=300, bbox_inches='tight')
    print(f"Average loss comparison plot saved to: {plot_path_avg}")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_dir / 'comprehensive_comparison.json'}")
    print(f"Plots saved to: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

