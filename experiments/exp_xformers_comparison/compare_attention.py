"""
Simple experiment comparing standard attention vs xformers memory-efficient attention
"""
import torch
import time
import json
from pathlib import Path
from dataclasses import asdict
from configs.moe_config import MoEModelConfig
from models.moe_llm import MoEMinimalLLM
from data.loader import get_dataloaders
from training.trainer import train_model

def get_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0

def run_experiment(use_mem_efficient: bool, config: MoEModelConfig):
    """Run training with specified attention type"""
    print(f"\n{'='*60}")
    print(f"Running with {'Memory-Efficient' if use_mem_efficient else 'Standard'} Attention")
    print(f"{'='*60}\n")
    
    # Update config
    config.use_mem_efficient_attention = use_mem_efficient
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Initialize model
    model = MoEMinimalLLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get data
    train_loader, val_loader, vocab_size = get_dataloaders(
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        num_documents=config.num_documents,
        max_tokens=config.max_tokens
    )
    config.vocab_size = vocab_size
    
    # Train
    start_time = time.time()
    results = train_model(model, train_loader, val_loader, config)
    train_time = time.time() - start_time
    
    # Collect metrics
    peak_memory = get_memory_usage()
    final_loss = results['val_losses'][-1] if results['val_losses'] else None
    
    metrics = {
        'attention_type': 'memory_efficient' if use_mem_efficient else 'standard',
        'train_time_seconds': train_time,
        'peak_memory_mb': peak_memory,
        'final_val_loss': final_loss,
        'steps_completed': len(results.get('train_losses', [])),
        'val_losses': results.get('val_losses', []),
        'train_losses': results.get('train_losses', [])
    }
    
    print(f"\n{'='*60}")
    print(f"Results: {'Memory-Efficient' if use_mem_efficient else 'Standard'} Attention")
    print(f"{'='*60}")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Peak Memory: {peak_memory:.2f} MB")
    print(f"Final Val Loss: {final_loss:.4f}" if final_loss else "Final Val Loss: N/A")
    print(f"{'='*60}\n")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return metrics

def main():
    # Setup
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Base config - small and fast for quick comparison
    config = MoEModelConfig(
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        batch_size=24,
        max_steps=50,  # Quick experiment
        max_seq_len=512,
        num_documents=1000,
        max_tokens=250000,
        eval_every=10,
        eval_steps=20
    )
    
    # Run both experiments
    results = []
    
    # Standard attention
    standard_metrics = run_experiment(use_mem_efficient=False, config=config)
    results.append(standard_metrics)
    
    # Memory-efficient attention
    mem_eff_metrics = run_experiment(use_mem_efficient=True, config=config)
    results.append(mem_eff_metrics)
    
    # Save results
    output_file = results_dir / "comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results': results,
            'comparison': {
                'speedup': standard_metrics['train_time_seconds'] / mem_eff_metrics['train_time_seconds'],
                'memory_reduction_mb': standard_metrics['peak_memory_mb'] - mem_eff_metrics['peak_memory_mb'],
                'memory_reduction_percent': (1 - mem_eff_metrics['peak_memory_mb'] / standard_metrics['peak_memory_mb']) * 100
            }
        }, f, indent=2)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Speedup: {standard_metrics['train_time_seconds'] / mem_eff_metrics['train_time_seconds']:.2f}x")
    print(f"Memory Reduction: {standard_metrics['peak_memory_mb'] - mem_eff_metrics['peak_memory_mb']:.2f} MB")
    print(f"Memory Reduction: {(1 - mem_eff_metrics['peak_memory_mb'] / standard_metrics['peak_memory_mb']) * 100:.1f}%")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

