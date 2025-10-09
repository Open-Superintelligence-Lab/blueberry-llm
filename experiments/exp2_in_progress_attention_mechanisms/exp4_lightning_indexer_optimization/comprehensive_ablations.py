#!/usr/bin/env python3
"""
Comprehensive Ablation Studies for Experiment 4

This module implements extensive ablation studies to systematically test
all optimization strategies and their combinations for Lightning Indexer optimization.

Ablation Categories:
1. Indexer Architecture Ablations
2. Attention Pattern Ablations  
3. Quantization Ablations
4. Adaptive Selection Ablations
5. Combined Strategy Ablations
6. Scaling Ablations
7. Hardware-Specific Ablations
"""

import os
import sys
import json
import time
import itertools
import argparse
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

from optimized_indexers import create_optimized_indexer
from efficient_patterns import create_efficient_pattern
from adaptive_selection import create_adaptive_selector
from quantization_utils import create_quantized_indexer
from exp4_models import create_optimized_model


@dataclass
class AblationConfig:
    """Configuration for comprehensive ablation studies"""
    # Model configuration
    vocab_size: int = 1000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 2048
    
    # Training configuration
    learning_rate: float = 1e-3
    batch_size: int = 16
    steps: int = 500  # Shorter for extensive ablations
    eval_every: int = 50
    
    # Sequence lengths for scaling analysis
    sequence_lengths: List[int] = None
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256, 512, 1024, 2048]


class SimpleTestDataset(Dataset):
    """Simple test dataset for ablation studies"""
    def __init__(self, seq_len: int = 128, num_samples: int = 1000, vocab_size: int = 1000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        targets = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {'input_ids': input_ids, 'targets': targets}


class ComprehensiveAblationRunner:
    """Runner for comprehensive ablation studies"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.results = {}
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create results directory
        os.makedirs('ablation_results', exist_ok=True)
    
    def define_ablation_studies(self) -> Dict[str, List[Dict]]:
        """Define all ablation studies to run"""
        
        ablations = {
            # 1. Indexer Architecture Ablations
            'indexer_architecture': [
                # Baseline configurations
                {'name': 'baseline_4h_64d', 'indexer_heads': 4, 'indexer_dim': 64},
                {'name': 'baseline_4h_32d', 'indexer_heads': 4, 'indexer_dim': 32},
                {'name': 'baseline_4h_16d', 'indexer_heads': 4, 'indexer_dim': 16},
                
                # Reduced head configurations
                {'name': 'optimized_3h_64d', 'indexer_heads': 3, 'indexer_dim': 64},
                {'name': 'optimized_3h_32d', 'indexer_heads': 3, 'indexer_dim': 32},
                {'name': 'optimized_2h_64d', 'indexer_heads': 2, 'indexer_dim': 64},
                {'name': 'optimized_2h_32d', 'indexer_heads': 2, 'indexer_dim': 32},
                {'name': 'optimized_2h_16d', 'indexer_heads': 2, 'indexer_dim': 16},
                
                # Minimal configurations
                {'name': 'minimal_1h_32d', 'indexer_heads': 1, 'indexer_dim': 32},
                {'name': 'minimal_1h_16d', 'indexer_heads': 1, 'indexer_dim': 16},
                {'name': 'minimal_1h_8d', 'indexer_heads': 1, 'indexer_dim': 8},
                {'name': 'minimal_1h_4d', 'indexer_heads': 1, 'indexer_dim': 4},
                
                # Ultra-minimal configurations
                {'name': 'ultra_1h_2d', 'indexer_heads': 1, 'indexer_dim': 2},
                {'name': 'ultra_1h_1d', 'indexer_heads': 1, 'indexer_dim': 1},
            ],
            
            # 2. Attention Pattern Ablations
            'attention_patterns': [
                # Local + Global patterns
                {'name': 'local_global_16_32', 'pattern': 'local_global', 'local_window': 16, 'global_k': 32},
                {'name': 'local_global_32_64', 'pattern': 'local_global', 'local_window': 32, 'global_k': 64},
                {'name': 'local_global_64_128', 'pattern': 'local_global', 'local_window': 64, 'global_k': 128},
                
                # Sliding window patterns
                {'name': 'sliding_32_16', 'pattern': 'sliding_window', 'window_size': 32, 'stride': 16},
                {'name': 'sliding_64_32', 'pattern': 'sliding_window', 'window_size': 64, 'stride': 32},
                {'name': 'sliding_128_64', 'pattern': 'sliding_window', 'window_size': 128, 'stride': 64},
                
                # Hierarchical patterns
                {'name': 'hierarchical_8_32_16', 'pattern': 'hierarchical', 'local_window': 8, 'medium_window': 32, 'global_k': 16},
                {'name': 'hierarchical_16_64_32', 'pattern': 'hierarchical', 'local_window': 16, 'medium_window': 64, 'global_k': 32},
                {'name': 'hierarchical_32_128_64', 'pattern': 'hierarchical', 'local_window': 32, 'medium_window': 128, 'global_k': 64},
                
                # Strided patterns
                {'name': 'strided_2', 'pattern': 'strided', 'stride': 2},
                {'name': 'strided_4', 'pattern': 'strided', 'stride': 4},
                {'name': 'strided_8', 'pattern': 'strided', 'stride': 8},
            ],
            
            # 3. Quantization Ablations
            'quantization': [
                # FP16 variants
                {'name': 'fp16_baseline', 'quantization': 'fp16', 'indexer_heads': 4, 'indexer_dim': 64},
                {'name': 'fp16_optimized', 'quantization': 'fp16', 'indexer_heads': 2, 'indexer_dim': 32},
                {'name': 'fp16_minimal', 'quantization': 'fp16', 'indexer_heads': 1, 'indexer_dim': 16},
                
                # Mixed precision variants
                {'name': 'mixed_baseline', 'quantization': 'mixed_precision', 'indexer_heads': 4, 'indexer_dim': 64},
                {'name': 'mixed_optimized', 'quantization': 'mixed_precision', 'indexer_heads': 2, 'indexer_dim': 32},
                {'name': 'mixed_minimal', 'quantization': 'mixed_precision', 'indexer_heads': 1, 'indexer_dim': 16},
                
                # INT8 variants
                {'name': 'int8_baseline', 'quantization': 'int8', 'indexer_heads': 4, 'indexer_dim': 64},
                {'name': 'int8_optimized', 'quantization': 'int8', 'indexer_heads': 2, 'indexer_dim': 32},
                {'name': 'int8_minimal', 'quantization': 'int8', 'indexer_heads': 1, 'indexer_dim': 16},
                
                # INT4 variants
                {'name': 'int4_baseline', 'quantization': 'int4', 'indexer_heads': 4, 'indexer_dim': 64},
                {'name': 'int4_optimized', 'quantization': 'int4', 'indexer_heads': 2, 'indexer_dim': 32},
                {'name': 'int4_minimal', 'quantization': 'int4', 'indexer_heads': 1, 'indexer_dim': 16},
            ],
            
            # 4. Adaptive Selection Ablations
            'adaptive_selection': [
                # Fixed ratio variants
                {'name': 'fixed_10', 'selector': 'fixed_ratio', 'ratio': 0.10},
                {'name': 'fixed_25', 'selector': 'fixed_ratio', 'ratio': 0.25},
                {'name': 'fixed_50', 'selector': 'fixed_ratio', 'ratio': 0.50},
                {'name': 'fixed_75', 'selector': 'fixed_ratio', 'ratio': 0.75},
                {'name': 'fixed_90', 'selector': 'fixed_ratio', 'ratio': 0.90},
                
                # Progressive variants
                {'name': 'progressive_linear', 'selector': 'progressive', 'progression_type': 'linear'},
                {'name': 'progressive_exp', 'selector': 'progressive', 'progression_type': 'exponential'},
                {'name': 'progressive_cosine', 'selector': 'progressive', 'progression_type': 'cosine'},
                
                # Adaptive variants
                {'name': 'adaptive_entropy', 'selector': 'adaptive', 'adaptation_strategy': 'entropy'},
                {'name': 'adaptive_position', 'selector': 'adaptive', 'adaptation_strategy': 'position'},
                {'name': 'adaptive_dynamic', 'selector': 'adaptive', 'adaptation_strategy': 'dynamic'},
            ],
            
            # 5. Combined Strategy Ablations
            'combined_strategies': [
                # Indexer + Pattern combinations
                {'name': 'opt2h32d_localglobal', 'indexer_heads': 2, 'indexer_dim': 32, 'pattern': 'local_global', 'local_window': 32, 'global_k': 64},
                {'name': 'opt2h32d_sliding', 'indexer_heads': 2, 'indexer_dim': 32, 'pattern': 'sliding_window', 'window_size': 64, 'stride': 32},
                {'name': 'min1h16d_localglobal', 'indexer_heads': 1, 'indexer_dim': 16, 'pattern': 'local_global', 'local_window': 16, 'global_k': 32},
                
                # Indexer + Quantization combinations
                {'name': 'opt2h32d_fp16', 'indexer_heads': 2, 'indexer_dim': 32, 'quantization': 'fp16'},
                {'name': 'opt2h32d_int8', 'indexer_heads': 2, 'indexer_dim': 32, 'quantization': 'int8'},
                {'name': 'min1h16d_fp16', 'indexer_heads': 1, 'indexer_dim': 16, 'quantization': 'fp16'},
                
                # Pattern + Selection combinations
                {'name': 'localglobal_fixed25', 'pattern': 'local_global', 'local_window': 32, 'global_k': 64, 'selector': 'fixed_ratio', 'ratio': 0.25},
                {'name': 'sliding_progressive', 'pattern': 'sliding_window', 'window_size': 64, 'stride': 32, 'selector': 'progressive'},
                
                # Triple combinations
                {'name': 'opt2h32d_localglobal_fp16', 'indexer_heads': 2, 'indexer_dim': 32, 'pattern': 'local_global', 'local_window': 32, 'global_k': 64, 'quantization': 'fp16'},
                {'name': 'min1h16d_sliding_fixed25', 'indexer_heads': 1, 'indexer_dim': 16, 'pattern': 'sliding_window', 'window_size': 32, 'stride': 16, 'selector': 'fixed_ratio', 'ratio': 0.25},
            ],
            
            # 6. Scaling Ablations
            'scaling_analysis': [
                # Model size scaling
                {'name': 'tiny_128d_4h_2l', 'd_model': 128, 'n_heads': 4, 'n_layers': 2},
                {'name': 'small_256d_8h_4l', 'd_model': 256, 'n_heads': 8, 'n_layers': 4},
                {'name': 'medium_512d_16h_8l', 'd_model': 512, 'n_heads': 16, 'n_layers': 8},
                {'name': 'large_1024d_32h_12l', 'd_model': 1024, 'n_heads': 32, 'n_layers': 12},
                
                # Sequence length scaling (tested separately)
                {'name': 'seq_scaling', 'test_sequence_lengths': True},
            ],
            
            # 7. Hardware-Specific Ablations
            'hardware_optimization': [
                # Memory optimization
                {'name': 'memory_opt_gradient_checkpointing', 'gradient_checkpointing': True},
                {'name': 'memory_opt_activation_offload', 'activation_offload': True},
                
                # Compute optimization
                {'name': 'compute_opt_flash_attention', 'use_flash_attention': True},
                {'name': 'compute_opt_tensor_parallelism', 'tensor_parallel_size': 2},
            ],
        }
        
        return ablations
    
    def run_single_ablation(
        self,
        ablation_name: str,
        ablation_config: Dict[str, Any],
        seq_len: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Dict[str, Any]:
        """Run a single ablation experiment"""
        
        try:
            print(f"\n{'='*80}")
            print(f"Running ablation: {ablation_name} (seq_len={seq_len})")
            print(f"Config: {ablation_config}")
            print(f"{'='*80}")
            
            # Create model with ablation configuration
            model = self._create_ablation_model(ablation_config).to(device)
            
            # Create dataset
            dataset = SimpleTestDataset(seq_len=seq_len, num_samples=1000, vocab_size=self.config.vocab_size)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            # Create optimizer
            optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            model.train()
            training_times = []
            losses = []
            
            start_time = time.time()
            
            for step, batch in enumerate(tqdm(dataloader, desc=f"Training {ablation_name}")):
                if step >= self.config.steps:
                    break
                
                step_start = time.time()
                
                # Forward pass
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                
                logits = model(input_ids)
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step_time = time.time() - step_start
                training_times.append(step_time)
                losses.append(loss.item())
                
                if step % self.config.eval_every == 0:
                    print(f"Step {step}: Loss = {loss.item():.4f}, Time = {step_time:.3f}s")
            
            total_time = time.time() - start_time
            
            # Evaluation
            model.eval()
            eval_losses = []
            eval_accuracies = []
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    targets = batch['targets'].to(device)
                    
                    logits = model(input_ids)
                    
                    # Compute loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-1
                    )
                    eval_losses.append(loss.item())
                    
                    # Compute accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    mask = targets != -1
                    accuracy = (predictions[mask] == targets[mask]).float().mean()
                    eval_accuracies.append(accuracy.item())
            
            # Calculate metrics
            avg_training_time = np.mean(training_times)
            final_loss = np.mean(eval_losses)
            final_accuracy = np.mean(eval_accuracies)
            perplexity = np.exp(final_loss)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            indexer_params = self._count_indexer_params(model)
            
            # Memory usage
            if device == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
            else:
                memory_allocated = memory_reserved = 0
            
            results = {
                'ablation_name': ablation_name,
                'ablation_config': ablation_config,
                'sequence_length': seq_len,
                'final_loss': final_loss,
                'final_accuracy': final_accuracy,
                'perplexity': perplexity,
                'avg_training_time': avg_training_time,
                'total_training_time': total_time,
                'total_params': total_params,
                'indexer_params': indexer_params,
                'indexer_param_ratio': indexer_params / total_params if total_params > 0 else 0,
                'memory_allocated_mb': memory_allocated,
                'memory_reserved_mb': memory_reserved,
                'training_losses': losses,
                'training_times': training_times
            }
            
            print(f"\nResults for {ablation_name} (seq_len={seq_len}):")
            print(f"  Final Loss: {final_loss:.4f}")
            print(f"  Final Accuracy: {final_accuracy:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Avg Training Time: {avg_training_time:.3f}s")
            print(f"  Total Params: {total_params:,}")
            print(f"  Indexer Params: {indexer_params:,} ({indexer_params/total_params*100:.1f}%)")
            
            return results
            
        except Exception as e:
            print(f"Error running {ablation_name} with seq_len {seq_len}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'ablation_name': ablation_name,
                'ablation_config': ablation_config,
                'sequence_length': seq_len,
                'error': str(e)
            }
    
    def _create_ablation_model(self, ablation_config: Dict[str, Any]):
        """Create model based on ablation configuration"""
        
        # Extract model configuration
        model_config = {
            'vocab_size': self.config.vocab_size,
            'd_model': ablation_config.get('d_model', self.config.d_model),
            'n_heads': ablation_config.get('n_heads', self.config.n_heads),
            'n_layers': ablation_config.get('n_layers', self.config.n_layers),
            'd_ff': ablation_config.get('d_ff', self.config.d_ff),
            'max_seq_len': self.config.max_seq_len,
        }
        
        # Create strategy kwargs
        strategy_kwargs = {}
        
        # Indexer configuration
        if 'indexer_heads' in ablation_config:
            strategy_kwargs['indexer_heads'] = ablation_config['indexer_heads']
        if 'indexer_dim' in ablation_config:
            strategy_kwargs['indexer_dim'] = ablation_config['indexer_dim']
        
        # Pattern configuration
        if 'pattern' in ablation_config:
            strategy_kwargs['pattern'] = ablation_config['pattern']
            if 'local_window' in ablation_config:
                strategy_kwargs['local_window'] = ablation_config['local_window']
            if 'global_k' in ablation_config:
                strategy_kwargs['global_k'] = ablation_config['global_k']
            if 'window_size' in ablation_config:
                strategy_kwargs['window_size'] = ablation_config['window_size']
            if 'stride' in ablation_config:
                strategy_kwargs['stride'] = ablation_config['stride']
        
        # Selection configuration
        if 'selector' in ablation_config:
            strategy_kwargs['selector'] = ablation_config['selector']
            if 'ratio' in ablation_config:
                strategy_kwargs['ratio'] = ablation_config['ratio']
        
        # Quantization configuration
        if 'quantization' in ablation_config:
            strategy_kwargs['quantization'] = ablation_config['quantization']
        
        # Determine strategy type
        if 'pattern' in ablation_config:
            strategy = ablation_config['pattern']
        elif 'quantization' in ablation_config:
            strategy = 'optimized'  # Use optimized as base for quantization
        elif 'selector' in ablation_config:
            strategy = 'optimized'  # Use optimized as base for selection
        else:
            strategy = 'optimized'  # Default to optimized
        
        return create_optimized_model(
            optimization_strategy=strategy,
            **model_config,
            strategy_kwargs=strategy_kwargs
        )
    
    def _count_indexer_params(self, model) -> int:
        """Count indexer parameters in model"""
        total = 0
        for layer in model.layers:
            if hasattr(layer, 'indexer'):
                total += sum(p.numel() for p in layer.indexer.parameters())
        return total
    
    def run_comprehensive_ablations(self, ablation_categories: List[str] = None):
        """Run comprehensive ablation studies"""
        
        if ablation_categories is None:
            ablation_categories = ['indexer_architecture', 'attention_patterns', 'quantization', 'adaptive_selection']
        
        print("Starting Comprehensive Ablation Studies")
        print(f"Categories: {ablation_categories}")
        print(f"Sequence Lengths: {self.config.sequence_lengths}")
        
        all_results = []
        ablation_definitions = self.define_ablation_studies()
        
        for category in ablation_categories:
            if category not in ablation_definitions:
                print(f"Warning: Unknown ablation category: {category}")
                continue
                
            print(f"\n{'='*100}")
            print(f"RUNNING ABLATION CATEGORY: {category.upper()}")
            print(f"{'='*100}")
            
            category_results = []
            ablations = ablation_definitions[category]
            
            for ablation_config in ablations:
                ablation_name = ablation_config['name']
                
                # Handle sequence scaling separately
                if ablation_name == 'seq_scaling':
                    for seq_len in self.config.sequence_lengths:
                        result = self.run_single_ablation(
                            f"{ablation_name}_seq{seq_len}",
                            ablation_config,
                            seq_len
                        )
                        category_results.append(result)
                        all_results.append(result)
                else:
                    # Test on representative sequence lengths
                    test_seq_lens = [64, 256, 1024] if len(self.config.sequence_lengths) > 3 else self.config.sequence_lengths
                    
                    for seq_len in test_seq_lens:
                        result = self.run_single_ablation(
                            f"{ablation_name}_seq{seq_len}",
                            ablation_config,
                            seq_len
                        )
                        category_results.append(result)
                        all_results.append(result)
                
                # Save intermediate results
                with open(f'ablation_results/{category}_intermediate.json', 'w') as f:
                    json.dump(category_results, f, indent=2)
            
            # Save category results
            with open(f'ablation_results/{category}_results.json', 'w') as f:
                json.dump(category_results, f, indent=2)
            
            print(f"\nCategory {category} complete. Results saved to ablation_results/{category}_results.json")
        
        # Save all results
        self.results = all_results
        with open('ablation_results/comprehensive_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create comprehensive analysis
        self.create_ablation_analysis()
        
        print(f"\n{'='*100}")
        print("COMPREHENSIVE ABLATION STUDIES COMPLETE!")
        print(f"{'='*100}")
        print("Results saved to ablation_results/")
    
    def create_ablation_analysis(self):
        """Create comprehensive analysis of ablation results"""
        
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(3, 3, figsize=(24, 18))
            fig.suptitle('Comprehensive Ablation Study Results', fontsize=20)
            
            # Group results by category
            categories = {}
            for result in self.results:
                if 'error' in result:
                    continue
                
                ablation_name = result['ablation_name']
                category = ablation_name.split('_')[0]  # Extract category prefix
                
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)
            
            # Plot 1: Parameter vs Performance scatter
            ax1 = axes[0, 0]
            for category, results in categories.items():
                params = [r['indexer_params'] for r in results]
                losses = [r['final_loss'] for r in results]
                ax1.scatter(params, losses, label=category, alpha=0.7, s=50)
            ax1.set_xlabel('Indexer Parameters')
            ax1.set_ylabel('Final Loss')
            ax1.set_title('Parameters vs Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Speed vs Quality scatter
            ax2 = axes[0, 1]
            for category, results in categories.items():
                times = [r['avg_training_time'] for r in results]
                losses = [r['final_loss'] for r in results]
                ax2.scatter(times, losses, label=category, alpha=0.7, s=50)
            ax2.set_xlabel('Training Time (s)')
            ax2.set_ylabel('Final Loss')
            ax2.set_title('Speed vs Quality')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Parameter efficiency
            ax3 = axes[0, 2]
            for category, results in categories.items():
                param_ratios = [r['indexer_param_ratio'] for r in results]
                losses = [r['final_loss'] for r in results]
                ax3.scatter(param_ratios, losses, label=category, alpha=0.7, s=50)
            ax3.set_xlabel('Indexer Parameter Ratio')
            ax3.set_ylabel('Final Loss')
            ax3.set_title('Parameter Efficiency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Sequence length scaling
            ax4 = axes[1, 0]
            seq_lengths = sorted(set(r['sequence_length'] for r in self.results))
            for category, results in categories.items():
                category_seq_results = {}
                for r in results:
                    seq_len = r['sequence_length']
                    if seq_len not in category_seq_results:
                        category_seq_results[seq_len] = []
                    category_seq_results[seq_len].append(r['final_loss'])
                
                avg_losses = []
                for seq_len in seq_lengths:
                    if seq_len in category_seq_results:
                        avg_loss = np.mean(category_seq_results[seq_len])
                        avg_losses.append(avg_loss)
                    else:
                        avg_losses.append(None)
                
                valid_indices = [i for i, x in enumerate(avg_losses) if x is not None]
                valid_losses = [avg_losses[i] for i in valid_indices]
                valid_seqs = [seq_lengths[i] for i in valid_indices]
                
                if valid_losses:
                    ax4.plot(valid_seqs, valid_losses, marker='o', label=category, linewidth=2)
            
            ax4.set_xlabel('Sequence Length')
            ax4.set_ylabel('Average Final Loss')
            ax4.set_title('Sequence Length Scaling')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Best configurations by category
            ax5 = axes[1, 1]
            best_configs = []
            for category, results in categories.items():
                # Find best result (lowest loss)
                best_result = min(results, key=lambda x: x['final_loss'])
                best_configs.append((category, best_result))
            
            categories_best = [bc[0] for bc in best_configs]
            losses_best = [bc[1]['final_loss'] for bc in best_configs]
            
            bars = ax5.bar(categories_best, losses_best, alpha=0.7)
            ax5.set_ylabel('Final Loss')
            ax5.set_title('Best Configuration by Category')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, losses_best):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(losses_best)*0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # Plot 6: Parameter reduction vs performance
            ax6 = axes[1, 2]
            baseline_params = max(r['indexer_params'] for r in self.results if 'baseline' in r['ablation_name'])
            
            param_reductions = []
            losses = []
            for result in self.results:
                if 'error' not in result:
                    reduction = (baseline_params - result['indexer_params']) / baseline_params * 100
                    param_reductions.append(reduction)
                    losses.append(result['final_loss'])
            
            ax6.scatter(param_reductions, losses, alpha=0.7, s=50)
            ax6.set_xlabel('Parameter Reduction (%)')
            ax6.set_ylabel('Final Loss')
            ax6.set_title('Parameter Reduction vs Performance')
            ax6.grid(True, alpha=0.3)
            
            # Plot 7: Speed improvement vs performance
            ax7 = axes[2, 0]
            baseline_time = max(r['avg_training_time'] for r in self.results if 'baseline' in r['ablation_name'])
            
            speed_improvements = []
            losses = []
            for result in self.results:
                if 'error' not in result:
                    improvement = (baseline_time - result['avg_training_time']) / baseline_time * 100
                    speed_improvements.append(improvement)
                    losses.append(result['final_loss'])
            
            ax7.scatter(speed_improvements, losses, alpha=0.7, s=50)
            ax7.set_xlabel('Speed Improvement (%)')
            ax7.set_ylabel('Final Loss')
            ax7.set_title('Speed Improvement vs Performance')
            ax7.grid(True, alpha=0.3)
            
            # Plot 8: Pareto frontier
            ax8 = axes[2, 1]
            for category, results in categories.items():
                times = [r['avg_training_time'] for r in results]
                losses = [r['final_loss'] for r in results]
                ax8.scatter(times, losses, label=category, alpha=0.7, s=50)
            
            ax8.set_xlabel('Training Time (s)')
            ax8.set_ylabel('Final Loss')
            ax8.set_title('Pareto Frontier (Speed vs Quality)')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            
            # Plot 9: Memory usage analysis
            ax9 = axes[2, 2]
            for category, results in categories.items():
                memory_usage = [r['memory_allocated_mb'] for r in results]
                losses = [r['final_loss'] for r in results]
                ax9.scatter(memory_usage, losses, label=category, alpha=0.7, s=50)
            
            ax9.set_xlabel('Memory Usage (MB)')
            ax9.set_ylabel('Final Loss')
            ax9.set_title('Memory Usage vs Performance')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ablation_results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Comprehensive analysis saved to ablation_results/comprehensive_analysis.png")
            
        except Exception as e:
            print(f"Error creating analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function for comprehensive ablation studies"""
    parser = argparse.ArgumentParser(description='Comprehensive Ablation Studies for Experiment 4')
    parser.add_argument('--categories', nargs='+', 
                       choices=['indexer_architecture', 'attention_patterns', 'quantization', 'adaptive_selection', 'combined_strategies', 'scaling_analysis', 'hardware_optimization'],
                       default=['indexer_architecture', 'attention_patterns', 'quantization', 'adaptive_selection'],
                       help='Ablation categories to run')
    parser.add_argument('--steps', type=int, default=500, help='Training steps per ablation')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seq-lens', nargs='+', type=int, help='Sequence lengths to test')
    
    args = parser.parse_args()
    
    # Create config
    config = AblationConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    if args.seq_lens:
        config.sequence_lengths = args.seq_lens
    
    # Run comprehensive ablations
    runner = ComprehensiveAblationRunner(config)
    runner.run_comprehensive_ablations(args.categories)


if __name__ == '__main__':
    main()
