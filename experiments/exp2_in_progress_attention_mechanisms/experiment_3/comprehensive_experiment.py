"""
Comprehensive Experiment 3 with Extensive Ablations

This script runs Experiment 3 with detailed ablation studies to thoroughly
test the adaptive sparsity system and understand its behavior.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/root/deepseek-sparse-attention-research')

# Import our modules
from adaptive_models import AdaptiveMoELLM, FixedSparseMoELLM
from simple_dataset import TinyStoriesDataset
from utils.helpers import set_seed, count_parameters


class ComprehensiveExperimentConfig:
    """Configuration for comprehensive experiment with ablations."""
    
    def __init__(self):
        # Model configuration
        self.d_model = 256  # Smaller for faster experiments
        self.n_layers = 4
        self.n_heads = 8
        self.d_ff = 1024
        self.vocab_size = 1000
        self.max_position_embeddings = 2048
        self.num_experts = 4
        self.top_k = 2
        self.dropout = 0.1
        
        # Training configuration
        self.learning_rate = 1e-3
        self.batch_size = 8
        self.steps = 500  # Reduced for faster experiments
        self.eval_every = 100
        self.warmup_steps = 50
        
        # Experiment configuration
        self.sequence_lengths = [64, 128, 256]  # Focus on key lengths
        self.random_seeds = [42, 123, 456]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results storage
        self.results_dir = 'comprehensive_results'
        os.makedirs(self.results_dir, exist_ok=True)


def create_model_config(config: ComprehensiveExperimentConfig) -> Dict:
    """Create model configuration dictionary."""
    return {
        'd_model': config.d_model,
        'n_layers': config.n_layers,
        'n_heads': config.n_heads,
        'd_ff': config.d_ff,
        'vocab_size': config.vocab_size,
        'max_position_embeddings': config.max_position_embeddings,
        'num_experts': config.num_experts,
        'top_k': config.top_k,
        'dropout': config.dropout
    }


def create_model(model_variant: Dict, config: ComprehensiveExperimentConfig):
    """Create model based on variant specification."""
    model_config = create_model_config(config)
    
    if model_variant['type'] == 'dense':
        return FixedSparseMoELLM(model_config, sparsity_ratio=1.0)
    elif model_variant['type'] == 'fixed_sparse':
        return FixedSparseMoELLM(model_config, sparsity_ratio=model_variant['sparsity_ratio'])
    elif model_variant['type'] == 'adaptive_sparse':
        return AdaptiveMoELLM(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_variant['type']}")


def train_model_simple(model, train_data, val_data, config: ComprehensiveExperimentConfig, 
                      model_name: str, seq_len: int, seed: int) -> Dict:
    """Simplified training function for faster experiments."""
    print(f"  Training {model_name} (seq_len={seq_len}, seed={seed})")
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Move model to device
    model = model.to(config.device)
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    step_times = []
    adaptive_stats_history = []
    
    model.train()
    start_time = time.time()
    
    # Create data iterators
    train_iter = iter(train_data)
    
    for step in range(config.steps):
        step_start_time = time.time()
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_data)
            batch = next(train_iter)
        
        input_ids = batch['input_ids'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record metrics
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        train_losses.append((step, loss.item()))
        
        # Validation
        if step % config.eval_every == 0:
            model.eval()
            val_loss, val_acc = evaluate_model_simple(model, val_data, config)
            val_losses.append((step, val_loss))
            val_accuracies.append((step, val_acc))
            
            # Adaptive stats (for adaptive models)
            if hasattr(model, 'get_adaptive_stats'):
                adaptive_stats = model.get_adaptive_stats()
                adaptive_stats_history.append((step, adaptive_stats))
            
            model.train()
    
    total_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    final_val_loss, final_val_acc = evaluate_model_simple(model, val_data, config)
    
    # Collect adaptive stats
    final_adaptive_stats = None
    if hasattr(model, 'get_adaptive_stats'):
        final_adaptive_stats = model.get_adaptive_stats()
    
    return {
        'model_name': model_name,
        'sequence_length': seq_len,
        'seed': seed,
        'total_time': total_time,
        'final_train_loss': train_losses[-1][1],
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'step_times': step_times,
        'adaptive_stats_history': adaptive_stats_history,
        'final_adaptive_stats': final_adaptive_stats,
        'model_parameters': count_parameters(model),
        'tokens_per_second': (len(train_losses) * config.batch_size * seq_len) / total_time
    }


def evaluate_model_simple(model, val_data, config: ComprehensiveExperimentConfig) -> Tuple[float, float]:
    """Simplified evaluation function."""
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        val_iter = iter(val_data)
        for _ in range(min(10, len(val_data))):  # Evaluate on subset
            try:
                batch = next(val_iter)
            except StopIteration:
                break
                
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # Calculate accuracy
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            predictions = torch.argmax(shift_logits, dim=-1)
            correct = (predictions == shift_labels).float()
            
            mask = (shift_labels != -100).float()
            total_correct += (correct * mask).sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / max(1, min(10, len(val_data)))
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy


def run_comprehensive_experiment():
    """Run comprehensive experiment with ablations."""
    print("="*80)
    print("COMPREHENSIVE EXPERIMENT 3: DYNAMIC SPARSITY WITH EXTENSIVE ABLATIONS")
    print("="*80)
    
    config = ComprehensiveExperimentConfig()
    
    print(f"Device: {config.device}")
    print(f"Sequence lengths: {config.sequence_lengths}")
    print(f"Model size: {config.d_model}d, {config.n_layers} layers, {config.n_heads} heads")
    print(f"Training steps: {config.steps}")
    
    all_results = {}
    
    # Model variants for comprehensive testing
    model_variants = [
        {'name': 'dense', 'type': 'dense', 'sparsity_ratio': 1.0, 'description': 'Full dense attention'},
        {'name': 'fixed_sparse_25', 'type': 'fixed_sparse', 'sparsity_ratio': 0.25, 'description': 'Fixed 25% sparsity'},
        {'name': 'fixed_sparse_50', 'type': 'fixed_sparse', 'sparsity_ratio': 0.5, 'description': 'Fixed 50% sparsity'},
        {'name': 'fixed_sparse_75', 'type': 'fixed_sparse', 'sparsity_ratio': 0.75, 'description': 'Fixed 75% sparsity'},
        {'name': 'adaptive_sparse', 'type': 'adaptive_sparse', 'sparsity_ratio': None, 'description': 'Adaptive sparsity'},
    ]
    
    for seq_len in config.sequence_lengths:
        print(f"\n{'='*60}")
        print(f"SEQUENCE LENGTH: {seq_len}")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = TinyStoriesDataset(
            seq_len=seq_len, 
            vocab_size=config.vocab_size,
            split='train',
            num_samples=1000
        )
        val_dataset = TinyStoriesDataset(
            seq_len=seq_len, 
            vocab_size=config.vocab_size,
            split='val',
            num_samples=200
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        seq_results = {}
        
        for model_variant in model_variants:
            model_name = model_variant['name']
            print(f"\nTesting {model_name}: {model_variant['description']}")
            
            variant_results = []
            
            # Test with multiple seeds
            for seed in config.random_seeds:
                try:
                    # Create model
                    model = create_model(model_variant, config)
                    
                    # Train model
                    result = train_model_simple(
                        model, train_loader, val_loader, config, 
                        model_name, seq_len, seed
                    )
                    
                    variant_results.append(result)
                    
                    # Clean up
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"    Error with seed {seed}: {e}")
                    continue
            
            seq_results[model_name] = variant_results
            print(f"  Completed {len(variant_results)} runs for {model_name}")
        
        all_results[seq_len] = seq_results
    
    # Analyze results
    print(f"\n{'='*60}")
    print("COMPREHENSIVE ANALYSIS")
    print(f"{'='*60}")
    
    analysis = analyze_comprehensive_results(all_results)
    
    # Save results
    save_comprehensive_results(all_results, analysis, config)
    
    # Print summary
    print_comprehensive_summary(analysis)
    
    return all_results, analysis


def analyze_comprehensive_results(all_results: Dict) -> Dict:
    """Analyze comprehensive results."""
    analysis = {}
    
    for seq_len, seq_results in all_results.items():
        print(f"\nSequence Length {seq_len}:")
        seq_analysis = {}
        
        for model_name, runs in seq_results.items():
            if not runs:
                continue
                
            # Aggregate across seeds
            final_losses = [run['final_val_loss'] for run in runs]
            final_accuracies = [run['final_val_accuracy'] for run in runs]
            training_times = [run['total_time'] for run in runs]
            tokens_per_second = [run['tokens_per_second'] for run in runs]
            
            seq_analysis[model_name] = {
                'val_loss_mean': np.mean(final_losses),
                'val_loss_std': np.std(final_losses),
                'val_accuracy_mean': np.mean(final_accuracies),
                'val_accuracy_std': np.std(final_accuracies),
                'training_time_mean': np.mean(training_times),
                'training_time_std': np.std(training_times),
                'tokens_per_second_mean': np.mean(tokens_per_second),
                'tokens_per_second_std': np.std(tokens_per_second),
                'n_runs': len(runs)
            }
            
            print(f"  {model_name}:")
            print(f"    Val Loss: {np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}")
            print(f"    Val Acc:  {np.mean(final_accuracies):.4f} ± {np.std(final_accuracies):.4f}")
            print(f"    Speed:    {np.mean(tokens_per_second):.0f} ± {np.std(tokens_per_second):.0f} tok/s")
        
        analysis[seq_len] = seq_analysis
    
    return analysis


def save_comprehensive_results(all_results: Dict, analysis: Dict, config: ComprehensiveExperimentConfig):
    """Save comprehensive results."""
    # Save detailed results
    results_file = os.path.join(config.results_dir, 'comprehensive_experiment_results.json')
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Convert all results
    serializable_results = {}
    for seq_len, seq_results in all_results.items():
        serializable_results[str(seq_len)] = {}
        for model_name, runs in seq_results.items():
            serializable_results[str(seq_len)][model_name] = []
            for run in runs:
                serializable_run = {}
                for key, value in run.items():
                    if isinstance(value, (list, tuple)):
                        serializable_run[key] = [convert_numpy(item) for item in value]
                    else:
                        serializable_run[key] = convert_numpy(value)
                serializable_results[str(seq_len)][model_name].append(serializable_run)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save analysis
    analysis_file = os.path.join(config.results_dir, 'comprehensive_analysis.json')
    
    serializable_analysis = {}
    for seq_len, seq_analysis in analysis.items():
        serializable_analysis[str(seq_len)] = {}
        for model_name, stats in seq_analysis.items():
            serializable_analysis[str(seq_len)][model_name] = {
                k: convert_numpy(v) for k, v in stats.items()
            }
    
    with open(analysis_file, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"\nResults saved to: {config.results_dir}/")


def print_comprehensive_summary(analysis: Dict):
    """Print comprehensive summary."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    print("\nKEY FINDINGS:")
    print("-" * 40)
    
    # Compare adaptive vs fixed sparsity
    for seq_len in sorted(analysis.keys()):
        if 'adaptive_sparse' in analysis[seq_len] and 'fixed_sparse_50' in analysis[seq_len]:
            adaptive_loss = analysis[seq_len]['adaptive_sparse']['val_loss_mean']
            fixed_loss = analysis[seq_len]['fixed_sparse_50']['val_loss_mean']
            improvement = (fixed_loss - adaptive_loss) / fixed_loss * 100
            
            adaptive_speed = analysis[seq_len]['adaptive_sparse']['tokens_per_second_mean']
            fixed_speed = analysis[seq_len]['fixed_sparse_50']['tokens_per_second_mean']
            speed_change = (adaptive_speed - fixed_speed) / fixed_speed * 100
            
            print(f"Seq Len {seq_len}:")
            print(f"  Loss improvement: {improvement:+.1f}%")
            print(f"  Speed change: {speed_change:+.1f}%")
    
    print(f"\n{'='*80}")


def run_adaptive_behavior_analysis():
    """Run detailed analysis of adaptive behavior."""
    print("\n" + "="*80)
    print("ADAPTIVE BEHAVIOR ANALYSIS")
    print("="*80)
    
    config = ComprehensiveExperimentConfig()
    model_config = create_model_config(config)
    
    # Test adaptive behavior across different sequence characteristics
    seq_lengths = [64, 128, 256]
    
    for seq_len in seq_lengths:
        print(f"\nAnalyzing adaptive behavior for sequence length {seq_len}:")
        
        # Create adaptive model
        model = AdaptiveMoELLM(model_config).to(config.device)
        
        # Test with different types of sequences
        test_cases = [
            ('uniform', torch.randn(1, seq_len, config.d_model)),  # Uniform random
            ('sparse', torch.randn(1, seq_len, config.d_model) * 0.1),  # Low variance
            ('dense', torch.randn(1, seq_len, config.d_model) * 10),  # High variance
        ]
        
        for case_name, hidden_states in test_cases:
            hidden_states = hidden_states.to(config.device)
            
            # Get adaptive characteristics
            with torch.no_grad():
                adaptive_k = model.layers[0].attention.sparsity_controller(hidden_states)
                characteristics = model.layers[0].attention.sparsity_controller.get_characteristics(hidden_states)
            
            print(f"  {case_name}:")
            print(f"    Adaptive k: {adaptive_k.item()}")
            print(f"    Length factor: {characteristics['length_factor'].item():.3f}")
            print(f"    Complexity factor: {characteristics['complexity_factor'].item():.3f}")
            print(f"    Entropy factor: {characteristics['entropy_factor'].item():.3f}")
            print(f"    Sparsity ratio: {characteristics['sparsity_ratio'].item():.3f}")
        
        del model
        torch.cuda.empty_cache()


def run_ablation_studies():
    """Run detailed ablation studies."""
    print("\n" + "="*80)
    print("ABLATION STUDIES")
    print("="*80)
    
    config = ComprehensiveExperimentConfig()
    model_config = create_model_config(config)
    
    # Test different sparsity ratios for fixed models
    print("\n1. Fixed Sparsity Ratio Ablation:")
    sparsity_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
    seq_len = 128
    
    for ratio in sparsity_ratios:
        model = FixedSparseMoELLM(model_config, sparsity_ratio=ratio)
        params = count_parameters(model)
        print(f"  Sparsity {ratio:.0%}: {params:,} parameters")
        del model
    
    # Test adaptive model parameter count
    print("\n2. Adaptive Model Parameter Analysis:")
    adaptive_model = AdaptiveMoELLM(model_config)
    total_params = count_parameters(adaptive_model)
    
    # Count parameters by component
    component_params = {}
    for name, param in adaptive_model.named_parameters():
        component = name.split('.')[0] + '.' + name.split('.')[1] if '.' in name else name
        if component not in component_params:
            component_params[component] = 0
        component_params[component] += param.numel()
    
    print(f"  Total parameters: {total_params:,}")
    for component, params in sorted(component_params.items()):
        percentage = params / total_params * 100
        print(f"  {component}: {params:,} ({percentage:.1f}%)")
    
    del adaptive_model


if __name__ == "__main__":
    # Run comprehensive experiment
    results, analysis = run_comprehensive_experiment()
    
    # Run additional analyses
    run_adaptive_behavior_analysis()
    run_ablation_studies()
    
    print("\n🎉 COMPREHENSIVE EXPERIMENT COMPLETED!")
    print("Check the 'comprehensive_results/' directory for detailed results.")
