"""
Ablation Study SPECIFICALLY for Recursive Reasoning Models
Tests different recursive reasoning configurations

All ablations run for 100 steps for quick iteration
"""

import torch
import torch.nn as nn
import sys
import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, replace
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp8_reasoning_architecture.config import get_base_reasoning_config
from experiments.exp8_reasoning_architecture.models import ReasoningModelWrapper
from data.loader import load_and_cache_data
from data.streaming_dataset import create_progressive_loaders
from utils.helpers import set_seed
from experiments.exp8_reasoning_architecture.run_ablations import QuickTrainer, plot_ablation_results


def run_ablation(name, config, recursive_config, train_loader, val_loader, device, max_steps=100):
    """Run a single ablation experiment"""
    print("\n" + "="*70)
    print(f"REASONING ABLATION: {name}")
    print("="*70)
    print(f"Config: LR={config.learning_rate}, Warmup={config.warmup_steps}")
    print(f"Recursive: H={recursive_config['H_cycles']}, L={recursive_config['L_cycles']}, "
          f"MaxSteps={recursive_config['halt_max_steps']}, ACT={recursive_config['use_act']}")
    
    try:
        # Add recursive config to main config
        config.recursive = recursive_config
        
        # Create model
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = ReasoningModelWrapper(config, use_recursive=True)
        model = model.to(device=device, dtype=dtype)
        
        # Train
        trainer = QuickTrainer(model, config, train_loader, val_loader, device, use_recursive=True)
        results = trainer.train(max_steps=max_steps)
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {name}")
        print(f"{'='*70}")
        print(f"Success: {results['success']}")
        print(f"Final Loss: {results['final_loss']:.4f}")
        print(f"Avg Loss: {results['avg_loss']:.4f}")
        print(f"Min Loss: {results['min_loss']:.4f}")
        print(f"Val Loss: {results.get('val_loss', float('inf')):.4f}")
        print(f"Val Perplexity: {results.get('val_perplexity', float('inf')):.2f}")
        print(f"Time: {results.get('time', 0):.1f}s")
        print(f"{'='*70}")
        
        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'final_loss': float('inf'),
            'avg_loss': float('inf'),
            'min_loss': float('inf'),
            'error': str(e),
        }


def main():
    """Run all reasoning ablations"""
    print("="*70)
    print("ABLATION STUDY: RECURSIVE REASONING MODELS")
    print("Testing different recursive configurations")
    print("All experiments run for 100 steps")
    print("="*70)
    
    # Base config
    base_config = get_base_reasoning_config()
    base_config.max_steps = 100
    base_config.warmup_steps = 20
    base_config.eval_interval = 50
    base_config.log_interval = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data (once for all ablations)
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    set_seed(base_config.seed)
    
    @dataclass
    class DataConfig:
        num_documents: int = base_config.num_documents
        max_tokens: int = base_config.max_tokens
        vocab_size: int = base_config.vocab_size
    
    data_config = DataConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    base_config.vocab_size = len(tokenizer)
    
    print(f"Vocabulary size: {base_config.vocab_size}")
    print(f"Total tokens: {len(tokens):,}")
    
    # Split tokens
    val_split_ratio = 0.1
    val_token_start = int(len(tokens) * (1 - val_split_ratio))
    train_tokens = tokens[:val_token_start]
    val_tokens = tokens[val_token_start:]
    
    # Create loaders
    train_loader, val_loader = create_progressive_loaders(
        train_tokens, val_tokens,
        base_config.max_seq_len, base_config.batch_size,
        None, None
    )
    
    print(f"Train windows: {len(train_loader):,}")
    print(f"Val windows: {len(val_loader):,}")
    
    # Define REASONING ablations
    ablations = {}
    
    # 1. Baseline recursive (H=2, L=2)
    ablations['R01_baseline'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 2. Single cycle (H=1, L=1) - minimal recursion
    ablations['R02_minimal'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 1,
            'L_cycles': 1,
            'halt_max_steps': 2,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 3. Deep recursion (H=3, L=3)
    ablations['R03_deep'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 3,
            'L_cycles': 3,
            'halt_max_steps': 5,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 4. More H cycles (H=3, L=1)
    ablations['R04_more_H'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 3,
            'L_cycles': 1,
            'halt_max_steps': 4,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 5. More L cycles (H=1, L=3)
    ablations['R05_more_L'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 1,
            'L_cycles': 3,
            'halt_max_steps': 4,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 6. No ACT (fixed cycles)
    ablations['R06_no_act'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.0,
            'use_act': False,
        }
    }
    
    # 7. Higher exploration
    ablations['R07_high_explore'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.3,
            'use_act': True,
        }
    }
    
    # 8. Low exploration
    ablations['R08_low_explore'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.01,
            'use_act': True,
        }
    }
    
    # 9. Extended max steps
    ablations['R09_extended_halt'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 7,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 10. Very low LR (for stability)
    config_low_lr = replace(base_config)
    config_low_lr.learning_rate = 1e-4
    ablations['R10_low_lr'] = {
        'config': config_low_lr,
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 11. Higher LR (like baseline winner)
    config_high_lr = replace(base_config)
    config_high_lr.learning_rate = 6e-4
    ablations['R11_high_lr'] = {
        'config': config_high_lr,
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 12. Medium LR
    config_med_lr = replace(base_config)
    config_med_lr.learning_rate = 2e-4
    ablations['R12_med_lr'] = {
        'config': config_med_lr,
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 13. No dropout + optimal cycles
    config_no_dropout = replace(base_config)
    config_no_dropout.dropout = 0.0
    config_no_dropout.learning_rate = 6e-4
    ablations['R13_optimized'] = {
        'config': config_no_dropout,
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 14. Asymmetric (H=2, L=1)
    ablations['R14_asym_H'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 2,
            'L_cycles': 1,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # 15. Asymmetric (H=1, L=2)
    ablations['R15_asym_L'] = {
        'config': replace(base_config),
        'recursive_config': {
            'H_cycles': 1,
            'L_cycles': 2,
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'use_act': True,
        }
    }
    
    # Run all ablations
    results_dir = Path(__file__).parent / "reasoning_ablation_results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    ablation_results = {}
    
    for name, setup in ablations.items():
        set_seed(base_config.seed)  # Reset seed for each ablation
        result = run_ablation(
            name,
            setup['config'],
            setup['recursive_config'],
            train_loader,
            val_loader,
            device,
            max_steps=100
        )
        ablation_results[name] = result
    
    # Save results
    results_file = results_dir / 'reasoning_ablation_results.json'
    
    # Convert results to serializable format
    serializable_results = {}
    for name, result in ablation_results.items():
        serializable_results[name] = {
            k: v for k, v in result.items() 
            if k not in ['losses', 'grad_norms']
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # Plot results
    plot_path = results_dir / 'reasoning_ablation_comparison.png'
    plot_ablation_results(ablation_results, plot_path)
    
    # Print summary
    print("\n" + "="*70)
    print("REASONING ABLATION SUMMARY")
    print("="*70)
    print(f"{'Name':<25} {'Final Loss':<12} {'Val Loss':<12} {'Time (s)':<10} {'Status':<10}")
    print("-"*70)
    
    for name, result in ablation_results.items():
        status = "✓ OK" if result.get('success', False) else "✗ FAIL"
        final_loss = result.get('final_loss', float('inf'))
        val_loss = result.get('val_loss', float('inf'))
        time_taken = result.get('time', 0)
        print(f"{name:<25} {final_loss:<12.4f} {val_loss:<12.4f} {time_taken:<10.1f} {status:<10}")
    
    print("="*70)
    
    # Find best ablation
    successful = {k: v for k, v in ablation_results.items() if v.get('success', False)}
    if successful:
        best_name = min(successful.keys(), key=lambda k: successful[k]['val_loss'])
        best_result = successful[best_name]
        print(f"\n🏆 BEST REASONING CONFIG: {best_name}")
        print(f"   Final Loss: {best_result['final_loss']:.4f}")
        print(f"   Val Loss: {best_result['val_loss']:.4f}")
        print(f"   Val Perplexity: {best_result['val_perplexity']:.2f}")
        print(f"   Time: {best_result['time']:.1f}s")
        
        # Compare to baseline
        baseline_val_loss = 7.08  # From previous ablations
        improvement = (baseline_val_loss - best_result['val_loss']) / baseline_val_loss * 100
        if improvement > 0:
            print(f"\n📊 Improvement over baseline: {improvement:.2f}% better")
        else:
            print(f"\n📊 Comparison to baseline: {-improvement:.2f}% worse (needs longer training)")
    
    print("\n✅ Reasoning ablation study completed!")


if __name__ == "__main__":
    main()

