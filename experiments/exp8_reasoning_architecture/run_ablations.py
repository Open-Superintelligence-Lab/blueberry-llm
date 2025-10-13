"""
Ablation Study for Reasoning Architecture
Tests different configurations to diagnose high loss issues

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

from experiments.exp8_reasoning_architecture.config import (
    ExperimentConfig,
    get_base_reasoning_config,
)
from experiments.exp8_reasoning_architecture.models import (
    ReasoningModelWrapper,
)
from data.loader import load_and_cache_data
from data.streaming_dataset import create_progressive_loaders
from utils.helpers import set_seed
from torch.utils.data import DataLoader


class QuickTrainer:
    """Minimal trainer for ablation studies"""
    
    def __init__(self, model, config, train_loader, val_loader, device, use_recursive=False):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_recursive = use_recursive
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        
        # Simple scheduler
        self.scheduler = self._create_scheduler()
        
        self.global_step = 0
        self.carry = None
        
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            # Avoid division by zero when warmup_steps == max_steps
            decay_steps = max(1, self.config.max_steps - self.config.warmup_steps)
            return max(0.1, (self.config.max_steps - step) / decay_steps)
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
        else:
            input_ids = batch.to(self.device)
        
        labels = input_ids.clone()
        
        # Forward pass
        if self.use_recursive:
            outputs = self.model(input_ids=input_ids, labels=labels, carry=self.carry)
            if 'carry' in outputs:
                self.carry = outputs['carry']
        else:
            outputs = self.model(input_ids=input_ids, labels=labels)
        
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            return None, {'nan_loss': True}
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        metrics = outputs.get('metrics', {}) if isinstance(outputs, dict) else {}
        metrics['grad_norm'] = grad_norm.item()
        
        return loss.item(), metrics
    
    @torch.no_grad()
    def evaluate(self, max_batches=10):
        """Quick evaluation"""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        eval_carry = None
        
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
            else:
                input_ids = batch.to(self.device)
            
            labels = input_ids.clone()
            
            if self.use_recursive:
                outputs = self.model(input_ids=input_ids, labels=labels, carry=eval_carry)
                if 'carry' in outputs:
                    eval_carry = outputs['carry']
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)
            
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        if total_tokens == 0:
            return {'loss': float('inf'), 'perplexity': float('inf')}
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {'loss': avg_loss, 'perplexity': perplexity}
    
    def train(self, max_steps=100):
        """Train for specified steps"""
        print(f"Training for {max_steps} steps...")
        
        losses = []
        grad_norms = []
        start_time = time.time()
        
        for batch in self.train_loader:
            if self.global_step >= max_steps:
                break
            
            loss, metrics = self.train_step(batch)
            
            if loss is None:
                print(f"⚠ NaN loss at step {self.global_step}")
                return {
                    'success': False,
                    'final_loss': float('inf'),
                    'avg_loss': float('inf'),
                    'min_loss': float('inf'),
                    'losses': losses,
                    'grad_norms': grad_norms,
                    'nan_encountered': True,
                }
            
            losses.append(loss)
            if 'grad_norm' in metrics:
                grad_norms.append(metrics['grad_norm'])
            
            self.global_step += 1
            
            if self.global_step % 20 == 0:
                recent_loss = sum(losses[-20:]) / min(20, len(losses))
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Step {self.global_step}/{max_steps} | Loss: {recent_loss:.4f} | LR: {lr:.6f}")
        
        # Final evaluation
        val_metrics = self.evaluate(max_batches=10)
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'final_loss': losses[-1] if losses else float('inf'),
            'avg_loss': sum(losses) / len(losses) if losses else float('inf'),
            'min_loss': min(losses) if losses else float('inf'),
            'val_loss': val_metrics['loss'],
            'val_perplexity': val_metrics['perplexity'],
            'losses': losses,
            'grad_norms': grad_norms,
            'time': elapsed,
            'nan_encountered': False,
        }


def run_ablation(name, config, use_recursive, train_loader, val_loader, device, max_steps=100):
    """Run a single ablation experiment"""
    print("\n" + "="*70)
    print(f"ABLATION: {name}")
    print("="*70)
    print(f"Config: LR={config.learning_rate}, Warmup={config.warmup_steps}, "
          f"GradClip={config.gradient_clip}, Recursive={use_recursive}")
    
    try:
        # Create model
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = ReasoningModelWrapper(config, use_recursive=use_recursive)
        model = model.to(device=device, dtype=dtype)
        
        # Train
        trainer = QuickTrainer(model, config, train_loader, val_loader, device, use_recursive)
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


def plot_ablation_results(ablation_results, save_path):
    """Plot comparison of all ablations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter successful runs
    successful = {k: v for k, v in ablation_results.items() if v.get('success', False)}
    
    if not successful:
        print("⚠ No successful ablations to plot")
        return
    
    # Plot 1: Training loss curves
    ax = axes[0, 0]
    for name, results in successful.items():
        if 'losses' in results and results['losses']:
            ax.plot(results['losses'], label=name, linewidth=2, alpha=0.8)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training Loss Curves', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_yscale('log')
    
    # Plot 2: Final metrics comparison
    ax = axes[0, 1]
    names = list(successful.keys())
    final_losses = [successful[n]['final_loss'] for n in names]
    colors = plt.cm.viridis(range(len(names)))
    bars = ax.barh(names, final_losses, color=colors)
    ax.set_xlabel('Final Loss', fontweight='bold')
    ax.set_title('Final Loss Comparison', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Validation loss
    ax = axes[1, 0]
    val_losses = [successful[n].get('val_loss', float('inf')) for n in names]
    bars = ax.barh(names, val_losses, color=colors)
    ax.set_xlabel('Validation Loss', fontweight='bold')
    ax.set_title('Validation Loss Comparison', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Gradient norms
    ax = axes[1, 1]
    for name, results in successful.items():
        if 'grad_norms' in results and results['grad_norms']:
            ax.plot(results['grad_norms'], label=name, linewidth=2, alpha=0.8)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontweight='bold')
    ax.set_title('Gradient Norms', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Ablation plots saved to: {save_path}")


def main():
    """Run all ablations"""
    print("="*70)
    print("ABLATION STUDY: Reasoning Architecture")
    print("All experiments run for 100 steps")
    print("="*70)
    
    # Base config
    base_config = get_base_reasoning_config()
    base_config.max_steps = 100
    base_config.warmup_steps = 20  # Ensure warmup < max_steps
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
    
    # Define ablations
    ablations = {}
    
    # 1. Baseline (MoE only)
    ablations['1_baseline'] = {
        'config': replace(base_config),
        'use_recursive': False,
    }
    
    # 2. Recursive (with hierarchical reasoning)
    ablations['2_recursive'] = {
        'config': replace(base_config),
        'use_recursive': True,
    }
    
    # 3. Lower learning rate
    config_low_lr = replace(base_config)
    config_low_lr.learning_rate = 1e-4
    ablations['3_low_lr'] = {
        'config': config_low_lr,
        'use_recursive': False,
    }
    
    # 4. Higher learning rate  
    config_high_lr = replace(base_config)
    config_high_lr.learning_rate = 6e-4
    ablations['4_high_lr'] = {
        'config': config_high_lr,
        'use_recursive': False,
    }
    
    # 5. Longer warmup
    config_long_warmup = replace(base_config)
    config_long_warmup.warmup_steps = 50
    ablations['5_long_warmup'] = {
        'config': config_long_warmup,
        'use_recursive': False,
    }
    
    # 6. No warmup
    config_no_warmup = replace(base_config)
    config_no_warmup.warmup_steps = 0
    ablations['6_no_warmup'] = {
        'config': config_no_warmup,
        'use_recursive': False,
    }
    
    # 7. Smaller model
    config_small = replace(base_config)
    config_small.hidden_size = 256
    config_small.num_hidden_layers = 4
    config_small.num_attention_heads = 4
    ablations['7_small_model'] = {
        'config': config_small,
        'use_recursive': False,
    }
    
    # 8. Fewer experts
    config_few_experts = replace(base_config)
    config_few_experts.num_experts = 4
    ablations['8_few_experts'] = {
        'config': config_few_experts,
        'use_recursive': False,
    }
    
    # 9. More top-k experts
    config_more_topk = replace(base_config)
    config_more_topk.expert_top_k = 4
    ablations['9_more_topk'] = {
        'config': config_more_topk,
        'use_recursive': False,
    }
    
    # 10. Lower gradient clip
    config_low_clip = replace(base_config)
    config_low_clip.gradient_clip = 0.5
    ablations['10_low_grad_clip'] = {
        'config': config_low_clip,
        'use_recursive': False,
    }
    
    # 11. Higher gradient clip
    config_high_clip = replace(base_config)
    config_high_clip.gradient_clip = 5.0
    ablations['11_high_grad_clip'] = {
        'config': config_high_clip,
        'use_recursive': False,
    }
    
    # 12. Recursive with very low LR
    config_rec_low_lr = replace(base_config)
    config_rec_low_lr.learning_rate = 1e-4
    ablations['12_recursive_vlow_lr'] = {
        'config': config_rec_low_lr,
        'use_recursive': True,
    }
    
    # 13. Very low LR baseline
    config_vlow_lr = replace(base_config)
    config_vlow_lr.learning_rate = 5e-5
    ablations['13_vlow_lr'] = {
        'config': config_vlow_lr,
        'use_recursive': False,
    }
    
    # 14. No dropout
    config_no_dropout = replace(base_config)
    config_no_dropout.dropout = 0.0
    ablations['14_no_dropout'] = {
        'config': config_no_dropout,
        'use_recursive': False,
    }
    
    # 15. Lower dropout
    config_low_dropout = replace(base_config)
    config_low_dropout.dropout = 0.05
    ablations['15_low_dropout'] = {
        'config': config_low_dropout,
        'use_recursive': False,
    }
    
    # Run all ablations
    results_dir = Path(__file__).parent / "ablation_results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    ablation_results = {}
    
    for name, setup in ablations.items():
        set_seed(base_config.seed)  # Reset seed for each ablation
        result = run_ablation(
            name,
            setup['config'],
            setup['use_recursive'],
            train_loader,
            val_loader,
            device,
            max_steps=100
        )
        ablation_results[name] = result
    
    # Save results
    results_file = results_dir / 'ablation_results.json'
    
    # Convert results to serializable format
    serializable_results = {}
    for name, result in ablation_results.items():
        serializable_results[name] = {
            k: v for k, v in result.items() 
            if k not in ['losses', 'grad_norms']  # Skip lists for JSON
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # Plot results
    plot_path = results_dir / 'ablation_comparison.png'
    plot_ablation_results(ablation_results, plot_path)
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(f"{'Name':<25} {'Final Loss':<12} {'Val Loss':<12} {'Status':<10}")
    print("-"*70)
    
    for name, result in ablation_results.items():
        status = "✓ OK" if result.get('success', False) else "✗ FAIL"
        final_loss = result.get('final_loss', float('inf'))
        val_loss = result.get('val_loss', float('inf'))
        print(f"{name:<25} {final_loss:<12.4f} {val_loss:<12.4f} {status:<10}")
    
    print("="*70)
    
    # Find best ablation
    successful = {k: v for k, v in ablation_results.items() if v.get('success', False)}
    if successful:
        best_name = min(successful.keys(), key=lambda k: successful[k]['val_loss'])
        best_result = successful[best_name]
        print(f"\n🏆 BEST ABLATION: {best_name}")
        print(f"   Final Loss: {best_result['final_loss']:.4f}")
        print(f"   Val Loss: {best_result['val_loss']:.4f}")
        print(f"   Val Perplexity: {best_result['val_perplexity']:.2f}")
    
    print("\n✅ Ablation study completed!")


if __name__ == "__main__":
    main()

