#!/usr/bin/env python3
"""
Experiment 4: Lightning Indexer Optimization

This script runs comprehensive experiments comparing various Lightning Indexer
optimization strategies to reduce computational overhead while maintaining
attention quality.

Optimization strategies tested:
1. Reduced complexity (fewer heads, smaller dimensions)
2. Efficient attention patterns (local+global, sliding window, etc.)
3. Quantization (FP16, mixed precision)
4. Adaptive k-value selection
5. Caching strategies

Usage:
    python run_experiment.py                    # Run all optimizations
    python run_experiment.py --strategy reduced_complexity  # Run specific strategy
    python run_experiment.py --seq-lens 64 128 256          # Custom sequence lengths
"""

import os
import sys
import json
import time
import argparse
import traceback
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import seaborn as sns
    sns_available = True
except ImportError:
    sns_available = False

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

# Import our modules
from optimized_indexers import (
    OptimizedLightningIndexer, MinimalLightningIndexer, UltraLightIndexer,
    CachedLightningIndexer, create_optimized_indexer, benchmark_indexer
)
from efficient_patterns import (
    LocalGlobalPattern, SlidingWindowPattern, HierarchicalPattern,
    create_efficient_pattern, analyze_pattern_coverage
)
from adaptive_selection import (
    AdaptiveKSelector, FixedRatioSelector, ProgressiveSelector,
    create_adaptive_selector, analyze_k_distribution
)

# Import existing components
from experiments.exp1_sparse_vs_classic_attention.sparse_attention import DeepSeekSparseAttention
from data.dataset import TextTokenDataset

# Create a simple test dataset
class SimpleTestDataset(Dataset):
    def __init__(self, seq_len: int = 128, num_samples: int = 1000, vocab_size: int = 1000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input and target sequences
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        targets = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {'input_ids': input_ids, 'targets': targets}


@dataclass
class ExperimentConfig:
    """Configuration for Experiment 4"""
    # Model configuration
    vocab_size: int = 1000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 1024
    
    # Training configuration
    learning_rate: float = 1e-3
    batch_size: int = 16
    steps: int = 1000
    eval_every: int = 100
    
    # Sequence lengths to test
    sequence_lengths: List[int] = None
    
    # Optimization strategies to test
    strategies: List[str] = None
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256, 512]
        
        if self.strategies is None:
            self.strategies = [
                'baseline',           # Original Lightning Indexer
                'optimized',          # 2 heads, 32 dims
                'minimal',            # 1 head, 16 dims
                'ultra_light',        # 1 head, 8 dims
                'local_global',       # Local + global pattern
                'sliding_window',     # Sliding window pattern
                'fixed_ratio_25',     # k = 25% of seq_len
                'fixed_ratio_50',     # k = 50% of seq_len
                'progressive',        # Progressive k during training
                'fp16_indexer',       # Half-precision indexer
            ]


class OptimizedSparseAttention(nn.Module):
    """
    Sparse attention with optimized Lightning Indexer variants
    """
    def __init__(
        self,
        config: ExperimentConfig,
        optimization_strategy: str = 'baseline',
        **strategy_kwargs
    ):
        super().__init__()
        self.config = config
        self.optimization_strategy = optimization_strategy
        
        # Standard components
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Rotary positional embeddings
        from torchtune.modules import RotaryPositionalEmbeddings
        self.rotary = RotaryPositionalEmbeddings(
            dim=config.d_model // config.n_heads,
            max_seq_len=config.max_seq_len
        )
        
        # Create optimized indexer based on strategy
        self.indexer = self._create_indexer(strategy_kwargs)
        
        # Create selector based on strategy
        self.selector = self._create_selector(strategy_kwargs)
        
        # Create efficient pattern if needed
        self.efficient_pattern = self._create_pattern(strategy_kwargs)
        
        self.dropout = nn.Dropout(0.1)
        self.use_sparse = True
        
    def _create_indexer(self, kwargs):
        """Create indexer based on optimization strategy"""
        if self.optimization_strategy == 'baseline':
            # Original Lightning Indexer (4 heads, 64 dims)
            return OptimizedLightningIndexer(
                d_model=self.config.d_model,
                indexer_heads=4,
                indexer_dim=64,
                use_fp16=False
            )
        elif self.optimization_strategy == 'optimized':
            # Optimized indexer (2 heads, 32 dims)
            return OptimizedLightningIndexer(
                d_model=self.config.d_model,
                indexer_heads=2,
                indexer_dim=32,
                use_fp16=kwargs.get('use_fp16', False)
            )
        elif self.optimization_strategy == 'minimal':
            # Minimal indexer (1 head, 16 dims)
            return MinimalLightningIndexer(
                d_model=self.config.d_model,
                indexer_dim=16,
                use_fp16=kwargs.get('use_fp16', False)
            )
        elif self.optimization_strategy == 'ultra_light':
            # Ultra-light indexer (1 head, 8 dims)
            return UltraLightIndexer(
                d_model=self.config.d_model,
                indexer_dim=8,
                use_fp16=True
            )
        elif self.optimization_strategy == 'fp16_indexer':
            # FP16 indexer (original config with FP16)
            return OptimizedLightningIndexer(
                d_model=self.config.d_model,
                indexer_heads=4,
                indexer_dim=64,
                use_fp16=True
            )
        else:
            # Default to optimized
            return OptimizedLightningIndexer(
                d_model=self.config.d_model,
                indexer_heads=2,
                indexer_dim=32
            )
    
    def _create_selector(self, kwargs):
        """Create selector based on optimization strategy"""
        if 'fixed_ratio' in self.optimization_strategy:
            ratio = 0.25 if '25' in self.optimization_strategy else 0.5
            return FixedRatioSelector(ratio=ratio)
        elif self.optimization_strategy == 'progressive':
            return ProgressiveSelector(start_k=16, end_k=256, max_steps=self.config.steps)
        else:
            # Default selector (from original sparse attention)
            from experiments.exp1_sparse_vs_classic_attention.sparse_attention import TopKTokenSelector
            return TopKTokenSelector()
    
    def _create_pattern(self, kwargs):
        """Create efficient pattern if needed"""
        if self.optimization_strategy == 'local_global':
            return LocalGlobalPattern(
                local_window=kwargs.get('local_window', 32),
                global_k=kwargs.get('global_k', 64),
                d_model=self.config.d_model
            )
        elif self.optimization_strategy == 'sliding_window':
            return SlidingWindowPattern(
                window_size=kwargs.get('window_size', 64),
                stride=kwargs.get('stride', 32),
                d_model=self.config.d_model
            )
        else:
            return None
    
    def forward(self, x: torch.Tensor, return_index_scores: bool = False):
        """Forward pass with optimized sparse attention"""
        batch_size, seq_len, _ = x.shape
        
        # Standard QKV computation
        QKV = self.qkv(x)
        Q, K, V = QKV.split(self.config.d_model, dim=-1)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.config.n_heads, self.config.d_model // self.config.n_heads)
        K = K.reshape(batch_size, seq_len, self.config.n_heads, self.config.d_model // self.config.n_heads)
        V = V.reshape(batch_size, seq_len, self.config.n_heads, self.config.d_model // self.config.n_heads)
        
        # Apply rotary positional embeddings
        Q = self.rotary(Q)
        K = self.rotary(K)
        
        # Compute index scores
        index_scores = self.indexer(x)
        
        if self.use_sparse:
            if self.efficient_pattern:
                # Use efficient pattern instead of selector
                attention_mask = self.efficient_pattern(x)
                
                # Apply attention with pattern mask
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                    attn_mask=attention_mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
            else:
                # Use selector-based sparse attention
                # Check if selector is adaptive (takes x) or original (takes only index_scores)
                if hasattr(self.selector, 'adaptation_strategy') or hasattr(self.selector, 'ratio'):
                    # Adaptive selector
                    top_k_mask, k_values = self.selector(x, index_scores, training_step=None)
                else:
                    # Original selector
                    top_k_mask, top_k_indices = self.selector(index_scores, apply_causal_mask=True)
                    k_values = torch.full((batch_size, seq_len), self.selector.top_k, dtype=torch.long, device=x.device)
                
                # Create attention mask from top-k selection
                attn_mask = torch.zeros(
                    batch_size, 1, seq_len, seq_len,
                    device=x.device, dtype=Q.dtype
                )
                attn_mask = attn_mask.masked_fill(~top_k_mask.unsqueeze(1), float('-inf'))
                
                # Apply sparse attention
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
        else:
            # Dense attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.config.d_model)
        output = self.w_o(attn_output)
        
        if return_index_scores:
            return output, index_scores
        return output, None


class OptimizedMoELLM(nn.Module):
    """
    Mixture of Experts LLM with optimized sparse attention
    """
    def __init__(self, config: ExperimentConfig, optimization_strategy: str = 'baseline'):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer layers with optimized sparse attention
        self.layers = nn.ModuleList([
            OptimizedSparseAttention(config, optimization_strategy)
            for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor):
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Transformer layers
        for layer in self.layers:
            layer_output = layer(hidden_states)
            if isinstance(layer_output, tuple):
                hidden_states = hidden_states + layer_output[0]  # Take first element (hidden states)
            else:
                hidden_states = hidden_states + layer_output
        
        # Output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits


class ExperimentRunner:
    """Main experiment runner for Lightning Indexer optimization"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def run_single_experiment(
        self, 
        strategy: str, 
        seq_len: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Dict[str, Any]:
        """Run single experiment for one strategy and sequence length"""
        print(f"\n{'='*60}")
        print(f"Running {strategy} with sequence length {seq_len}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = OptimizedMoELLM(self.config, strategy).to(device)
            
            # Create dataset
            dataset = SimpleTestDataset(seq_len=seq_len, num_samples=1000, vocab_size=1000)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            # Create optimizer
            optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            model.train()
            training_times = []
            losses = []
            
            start_time = time.time()
            
            for step, batch in enumerate(tqdm(dataloader, desc=f"Training {strategy}")):
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
            indexer_params = sum(p.numel() for p in model.layers[0].indexer.parameters())
            
            # Memory usage
            if device == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
            else:
                memory_allocated = memory_reserved = 0
            
            results = {
                'strategy': strategy,
                'sequence_length': seq_len,
                'final_loss': final_loss,
                'final_accuracy': final_accuracy,
                'perplexity': perplexity,
                'avg_training_time': avg_training_time,
                'total_training_time': total_time,
                'total_params': total_params,
                'indexer_params': indexer_params,
                'indexer_param_ratio': indexer_params / total_params,
                'memory_allocated_mb': memory_allocated,
                'memory_reserved_mb': memory_reserved,
                'training_losses': losses,
                'training_times': training_times
            }
            
            print(f"\nResults for {strategy} (seq_len={seq_len}):")
            print(f"  Final Loss: {final_loss:.4f}")
            print(f"  Final Accuracy: {final_accuracy:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Avg Training Time: {avg_training_time:.3f}s")
            print(f"  Total Params: {total_params:,}")
            print(f"  Indexer Params: {indexer_params:,} ({indexer_params/total_params*100:.1f}%)")
            
            return results
            
        except Exception as e:
            print(f"Error running {strategy} with seq_len {seq_len}: {e}")
            traceback.print_exc()
            return {
                'strategy': strategy,
                'sequence_length': seq_len,
                'error': str(e)
            }
    
    def run_full_experiment(self):
        """Run the complete experiment with all strategies and sequence lengths"""
        print("Starting Experiment 4: Lightning Indexer Optimization")
        print(f"Strategies: {self.config.strategies}")
        print(f"Sequence Lengths: {self.config.sequence_lengths}")
        
        all_results = []
        
        for strategy in self.config.strategies:
            for seq_len in self.config.sequence_lengths:
                result = self.run_single_experiment(strategy, seq_len)
                all_results.append(result)
                
                # Save intermediate results
                with open('results/intermediate_results.json', 'w') as f:
                    json.dump(all_results, f, indent=2)
        
        self.results = all_results
        self.save_results()
        self.create_visualizations()
        
        print("\n" + "="*60)
        print("Experiment 4 Complete!")
        print("="*60)
        print("Results saved to results/")
    
    def save_results(self):
        """Save experiment results"""
        # Save detailed results
        with open('results/detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary
        summary = self.create_summary()
        with open('results/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Results saved to results/detailed_results.json and results/summary.json")
    
    def create_summary(self) -> Dict[str, Any]:
        """Create summary of results"""
        # Group results by strategy
        strategy_results = {}
        
        for result in self.results:
            if 'error' in result:
                continue
                
            strategy = result['strategy']
            if strategy not in strategy_results:
                strategy_results[strategy] = {}
            
            seq_len = result['sequence_length']
            strategy_results[strategy][seq_len] = {
                'loss': result['final_loss'],
                'accuracy': result['final_accuracy'],
                'perplexity': result['perplexity'],
                'avg_time': result['avg_training_time'],
                'total_params': result['total_params'],
                'indexer_params': result['indexer_params'],
                'indexer_ratio': result['indexer_param_ratio']
            }
        
        # Calculate relative improvements vs baseline
        improvements = {}
        if 'baseline' in strategy_results:
            baseline = strategy_results['baseline']
            
            for strategy, results in strategy_results.items():
                if strategy == 'baseline':
                    continue
                
                improvements[strategy] = {}
                for seq_len in results:
                    if seq_len in baseline:
                        baseline_loss = baseline[seq_len]['loss']
                        baseline_time = baseline[seq_len]['avg_time']
                        
                        strategy_loss = results[seq_len]['loss']
                        strategy_time = results[seq_len]['avg_time']
                        
                        improvements[strategy][seq_len] = {
                            'loss_improvement': (baseline_loss - strategy_loss) / baseline_loss * 100,
                            'speed_improvement': (baseline_time - strategy_time) / baseline_time * 100,
                            'param_reduction': (baseline[seq_len]['indexer_params'] - results[seq_len]['indexer_params']) / baseline[seq_len]['indexer_params'] * 100
                        }
        
        return {
            'strategy_results': strategy_results,
            'improvements_vs_baseline': improvements,
            'config': {
                'strategies': self.config.strategies,
                'sequence_lengths': self.config.sequence_lengths,
                'steps': self.config.steps,
                'seed': self.config.seed
            }
        }
    
    def create_visualizations(self):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Set style
            plt.style.use('default')
            if sns_available:
                sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Experiment 4: Lightning Indexer Optimization Results', fontsize=16)
            
            # Extract data for plotting
            strategies = []
            seq_lengths = []
            losses = []
            accuracies = []
            times = []
            indexer_params = []
            
            for result in self.results:
                if 'error' not in result:
                    strategies.append(result['strategy'])
                    seq_lengths.append(result['sequence_length'])
                    losses.append(result['final_loss'])
                    accuracies.append(result['final_accuracy'])
                    times.append(result['avg_training_time'])
                    indexer_params.append(result['indexer_params'])
            
            # Plot 1: Loss vs Sequence Length
            ax1 = axes[0, 0]
            for strategy in set(strategies):
                strategy_data = [(seq_len, loss) for s, seq_len, loss in zip(strategies, seq_lengths, losses) if s == strategy]
                if strategy_data:
                    seq_lens, strategy_losses = zip(*strategy_data)
                    ax1.plot(seq_lens, strategy_losses, marker='o', label=strategy, linewidth=2)
            ax1.set_xlabel('Sequence Length')
            ax1.set_ylabel('Validation Loss')
            ax1.set_title('Loss vs Sequence Length')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Accuracy vs Sequence Length
            ax2 = axes[0, 1]
            for strategy in set(strategies):
                strategy_data = [(seq_len, acc) for s, seq_len, acc in zip(strategies, seq_lengths, accuracies) if s == strategy]
                if strategy_data:
                    seq_lens, strategy_accs = zip(*strategy_data)
                    ax2.plot(seq_lens, strategy_accs, marker='o', label=strategy, linewidth=2)
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Validation Accuracy')
            ax2.set_title('Accuracy vs Sequence Length')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Training Time vs Sequence Length
            ax3 = axes[0, 2]
            for strategy in set(strategies):
                strategy_data = [(seq_len, time) for s, seq_len, time in zip(strategies, seq_lengths, times) if s == strategy]
                if strategy_data:
                    seq_lens, strategy_times = zip(*strategy_data)
                    ax3.plot(seq_lens, strategy_times, marker='o', label=strategy, linewidth=2)
            ax3.set_xlabel('Sequence Length')
            ax3.set_ylabel('Average Training Time (s)')
            ax3.set_title('Training Time vs Sequence Length')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Indexer Parameters by Strategy
            ax4 = axes[1, 0]
            strategy_param_counts = {}
            for strategy, params in zip(strategies, indexer_params):
                if strategy not in strategy_param_counts:
                    strategy_param_counts[strategy] = []
                strategy_param_counts[strategy].append(params)
            
            strategy_names = list(strategy_param_counts.keys())
            param_means = [np.mean(strategy_param_counts[s]) for s in strategy_names]
            
            bars = ax4.bar(strategy_names, param_means, alpha=0.7)
            ax4.set_ylabel('Average Indexer Parameters')
            ax4.set_title('Indexer Parameter Count by Strategy')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, param_means):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_means)*0.01,
                        f'{value:,.0f}', ha='center', va='bottom')
            
            # Plot 5: Speed vs Quality Trade-off
            ax5 = axes[1, 1]
            for strategy in set(strategies):
                strategy_data = [(loss, time) for s, loss, time in zip(strategies, losses, times) if s == strategy]
                if strategy_data:
                    strategy_losses, strategy_times = zip(*strategy_data)
                    ax5.scatter(strategy_losses, strategy_times, label=strategy, s=100, alpha=0.7)
            ax5.set_xlabel('Validation Loss')
            ax5.set_ylabel('Average Training Time (s)')
            ax5.set_title('Speed vs Quality Trade-off')
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Relative Improvements vs Baseline
            ax6 = axes[1, 2]
            
            # Calculate improvements vs baseline
            baseline_times = {seq_len: time for s, seq_len, time in zip(strategies, seq_lengths, times) if s == 'baseline'}
            baseline_losses = {seq_len: loss for s, seq_len, loss in zip(strategies, seq_lengths, losses) if s == 'baseline'}
            
            improvements = []
            improvement_labels = []
            
            for strategy in set(strategies):
                if strategy == 'baseline':
                    continue
                    
                for seq_len in set(seq_lengths):
                    if seq_len in baseline_times and seq_len in baseline_losses:
                        strategy_data = [(s, sl, st, sloss) for s, sl, st, sloss in zip(strategies, seq_lengths, times, losses) if s == strategy and sl == seq_len]
                        if strategy_data:
                            _, _, strategy_time, strategy_loss = strategy_data[0]
                            
                            time_improvement = (baseline_times[seq_len] - strategy_time) / baseline_times[seq_len] * 100
                            loss_improvement = (baseline_losses[seq_len] - strategy_loss) / baseline_losses[seq_len] * 100
                            
                            improvements.append((time_improvement, loss_improvement))
                            improvement_labels.append(f"{strategy}\n(seq={seq_len})")
            
            if improvements:
                time_imps, loss_imps = zip(*improvements)
                ax6.scatter(time_imps, loss_imps, s=100, alpha=0.7)
                
                # Add labels
                for i, label in enumerate(improvement_labels):
                    ax6.annotate(label, (time_imps[i], loss_imps[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax6.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax6.set_xlabel('Speed Improvement (%)')
                ax6.set_ylabel('Quality Improvement (%)')
                ax6.set_title('Improvements vs Baseline')
                ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/optimization_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Visualizations saved to results/optimization_comparison.png")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            traceback.print_exc()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Experiment 4: Lightning Indexer Optimization')
    parser.add_argument('--strategies', nargs='+', help='Strategies to test')
    parser.add_argument('--seq-lens', nargs='+', type=int, help='Sequence lengths to test')
    parser.add_argument('--steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    if args.strategies:
        config.strategies = args.strategies
    if args.seq_lens:
        config.sequence_lengths = args.seq_lens
    
    # Run experiment
    runner = ExperimentRunner(config)
    runner.run_full_experiment()


if __name__ == '__main__':
    main()
