#!/usr/bin/env python3
"""
Specialized Ablation Studies for Specific Research Questions

This module implements focused ablation studies to answer specific research questions
about Lightning Indexer optimization.
"""

import os
import json
import itertools
from typing import Dict, List, Any
import torch
import numpy as np

from comprehensive_ablations import ComprehensiveAblationRunner, AblationConfig


class SpecializedAblationRunner(ComprehensiveAblationRunner):
    """Runner for specialized ablation studies"""
    
    def define_research_questions(self) -> Dict[str, Dict[str, Any]]:
        """Define specific research questions and their ablation studies"""
        
        research_questions = {
            # Research Question 1: What is the optimal indexer configuration?
            'optimal_indexer_config': {
                'question': 'What is the optimal Lightning Indexer configuration for different sequence lengths?',
                'hypothesis': 'Optimal configuration varies with sequence length: smaller configs for short sequences, larger for long sequences',
                'ablations': [
                    # Systematic sweep of head/dim combinations
                    {'name': 'h1d8', 'indexer_heads': 1, 'indexer_dim': 8},
                    {'name': 'h1d16', 'indexer_heads': 1, 'indexer_dim': 16},
                    {'name': 'h1d32', 'indexer_heads': 1, 'indexer_dim': 32},
                    {'name': 'h1d64', 'indexer_heads': 1, 'indexer_dim': 64},
                    {'name': 'h2d8', 'indexer_heads': 2, 'indexer_dim': 8},
                    {'name': 'h2d16', 'indexer_heads': 2, 'indexer_dim': 16},
                    {'name': 'h2d32', 'indexer_heads': 2, 'indexer_dim': 32},
                    {'name': 'h2d64', 'indexer_heads': 2, 'indexer_dim': 64},
                    {'name': 'h3d16', 'indexer_heads': 3, 'indexer_dim': 16},
                    {'name': 'h3d32', 'indexer_heads': 3, 'indexer_dim': 32},
                    {'name': 'h3d64', 'indexer_heads': 3, 'indexer_dim': 64},
                    {'name': 'h4d16', 'indexer_heads': 4, 'indexer_dim': 16},
                    {'name': 'h4d32', 'indexer_heads': 4, 'indexer_dim': 32},
                    {'name': 'h4d64', 'indexer_heads': 4, 'indexer_dim': 64},
                ],
                'sequence_lengths': [64, 128, 256, 512, 1024, 2048],
                'metrics': ['loss', 'accuracy', 'perplexity', 'training_time', 'parameters']
            },
            
            # Research Question 2: How do attention patterns affect performance?
            'attention_pattern_analysis': {
                'question': 'Which attention patterns provide the best speed-quality trade-offs?',
                'hypothesis': 'Local+Global patterns provide best trade-offs, hierarchical patterns scale best to long sequences',
                'ablations': [
                    # Local+Global variants
                    {'name': 'lg_16_32', 'pattern': 'local_global', 'local_window': 16, 'global_k': 32},
                    {'name': 'lg_32_64', 'pattern': 'local_global', 'local_window': 32, 'global_k': 64},
                    {'name': 'lg_64_128', 'pattern': 'local_global', 'local_window': 64, 'global_k': 128},
                    {'name': 'lg_128_256', 'pattern': 'local_global', 'local_window': 128, 'global_k': 256},
                    
                    # Sliding window variants
                    {'name': 'sw_32_16', 'pattern': 'sliding_window', 'window_size': 32, 'stride': 16},
                    {'name': 'sw_64_32', 'pattern': 'sliding_window', 'window_size': 64, 'stride': 32},
                    {'name': 'sw_128_64', 'pattern': 'sliding_window', 'window_size': 128, 'stride': 64},
                    {'name': 'sw_256_128', 'pattern': 'sliding_window', 'window_size': 256, 'stride': 128},
                    
                    # Hierarchical variants
                    {'name': 'hier_8_32_16', 'pattern': 'hierarchical', 'local_window': 8, 'medium_window': 32, 'global_k': 16},
                    {'name': 'hier_16_64_32', 'pattern': 'hierarchical', 'local_window': 16, 'medium_window': 64, 'global_k': 32},
                    {'name': 'hier_32_128_64', 'pattern': 'hierarchical', 'local_window': 32, 'medium_window': 128, 'global_k': 64},
                    {'name': 'hier_64_256_128', 'pattern': 'hierarchical', 'local_window': 64, 'medium_window': 256, 'global_k': 128},
                ],
                'sequence_lengths': [128, 512, 1024, 2048],
                'metrics': ['loss', 'accuracy', 'training_time', 'memory_usage', 'coverage']
            },
            
            # Research Question 3: What is the optimal k-value selection strategy?
            'k_value_optimization': {
                'question': 'What is the optimal k-value selection strategy for different sequence lengths?',
                'hypothesis': 'Fixed ratios work best for short sequences, adaptive strategies work better for long sequences',
                'ablations': [
                    # Fixed ratio variants
                    {'name': 'fixed_5', 'selector': 'fixed_ratio', 'ratio': 0.05},
                    {'name': 'fixed_10', 'selector': 'fixed_ratio', 'ratio': 0.10},
                    {'name': 'fixed_25', 'selector': 'fixed_ratio', 'ratio': 0.25},
                    {'name': 'fixed_50', 'selector': 'fixed_ratio', 'ratio': 0.50},
                    {'name': 'fixed_75', 'selector': 'fixed_ratio', 'ratio': 0.75},
                    {'name': 'fixed_90', 'selector': 'fixed_ratio', 'ratio': 0.90},
                    
                    # Progressive variants
                    {'name': 'prog_linear', 'selector': 'progressive', 'progression_type': 'linear', 'start_k': 16, 'end_k': 256},
                    {'name': 'prog_exp', 'selector': 'progressive', 'progression_type': 'exponential', 'start_k': 16, 'end_k': 256},
                    {'name': 'prog_cosine', 'selector': 'progressive', 'progression_type': 'cosine', 'start_k': 16, 'end_k': 256},
                    
                    # Adaptive variants
                    {'name': 'adaptive_entropy', 'selector': 'adaptive', 'adaptation_strategy': 'entropy'},
                    {'name': 'adaptive_position', 'selector': 'adaptive', 'adaptation_strategy': 'position'},
                    {'name': 'adaptive_dynamic', 'selector': 'adaptive', 'adaptation_strategy': 'dynamic'},
                ],
                'sequence_lengths': [128, 512, 1024, 2048],
                'metrics': ['loss', 'accuracy', 'k_distribution', 'adaptation_quality']
            },
            
            # Research Question 4: How effective is quantization?
            'quantization_effectiveness': {
                'question': 'How does quantization affect Lightning Indexer performance and efficiency?',
                'hypothesis': 'FP16 provides good speedup with minimal quality loss, INT8/INT4 provide more speedup but higher quality loss',
                'ablations': [
                    # Baseline configurations with different quantizations
                    {'name': 'fp32_baseline', 'quantization': None, 'indexer_heads': 4, 'indexer_dim': 64},
                    {'name': 'fp16_baseline', 'quantization': 'fp16', 'indexer_heads': 4, 'indexer_dim': 64},
                    {'name': 'int8_baseline', 'quantization': 'int8', 'indexer_heads': 4, 'indexer_dim': 64},
                    {'name': 'int4_baseline', 'quantization': 'int4', 'indexer_heads': 4, 'indexer_dim': 64},
                    
                    {'name': 'fp32_optimized', 'quantization': None, 'indexer_heads': 2, 'indexer_dim': 32},
                    {'name': 'fp16_optimized', 'quantization': 'fp16', 'indexer_heads': 2, 'indexer_dim': 32},
                    {'name': 'int8_optimized', 'quantization': 'int8', 'indexer_heads': 2, 'indexer_dim': 32},
                    {'name': 'int4_optimized', 'quantization': 'int4', 'indexer_heads': 2, 'indexer_dim': 32},
                    
                    {'name': 'fp32_minimal', 'quantization': None, 'indexer_heads': 1, 'indexer_dim': 16},
                    {'name': 'fp16_minimal', 'quantization': 'fp16', 'indexer_heads': 1, 'indexer_dim': 16},
                    {'name': 'int8_minimal', 'quantization': 'int8', 'indexer_heads': 1, 'indexer_dim': 16},
                    {'name': 'int4_minimal', 'quantization': 'int4', 'indexer_heads': 1, 'indexer_dim': 16},
                ],
                'sequence_lengths': [128, 512, 1024],
                'metrics': ['loss', 'accuracy', 'training_time', 'memory_usage', 'quantization_error']
            },
            
            # Research Question 5: How do optimizations combine?
            'optimization_combinations': {
                'question': 'Which combinations of optimizations provide the best overall performance?',
                'hypothesis': 'Best combinations depend on sequence length: minimal indexer + patterns for short sequences, optimized indexer + quantization for long sequences',
                'ablations': [
                    # Single optimizations
                    {'name': 'baseline', 'indexer_heads': 4, 'indexer_dim': 64},
                    {'name': 'opt_indexer', 'indexer_heads': 2, 'indexer_dim': 32},
                    {'name': 'min_indexer', 'indexer_heads': 1, 'indexer_dim': 16},
                    {'name': 'local_global', 'pattern': 'local_global', 'local_window': 32, 'global_k': 64},
                    {'name': 'fp16', 'quantization': 'fp16', 'indexer_heads': 2, 'indexer_dim': 32},
                    {'name': 'fixed_25', 'selector': 'fixed_ratio', 'ratio': 0.25},
                    
                    # Double combinations
                    {'name': 'opt_indexer_fp16', 'indexer_heads': 2, 'indexer_dim': 32, 'quantization': 'fp16'},
                    {'name': 'opt_indexer_localglobal', 'indexer_heads': 2, 'indexer_dim': 32, 'pattern': 'local_global', 'local_window': 32, 'global_k': 64},
                    {'name': 'opt_indexer_fixed25', 'indexer_heads': 2, 'indexer_dim': 32, 'selector': 'fixed_ratio', 'ratio': 0.25},
                    {'name': 'localglobal_fp16', 'pattern': 'local_global', 'local_window': 32, 'global_k': 64, 'quantization': 'fp16'},
                    {'name': 'localglobal_fixed25', 'pattern': 'local_global', 'local_window': 32, 'global_k': 64, 'selector': 'fixed_ratio', 'ratio': 0.25},
                    {'name': 'fp16_fixed25', 'quantization': 'fp16', 'indexer_heads': 2, 'indexer_dim': 32, 'selector': 'fixed_ratio', 'ratio': 0.25},
                    
                    # Triple combinations
                    {'name': 'opt_indexer_localglobal_fp16', 'indexer_heads': 2, 'indexer_dim': 32, 'pattern': 'local_global', 'local_window': 32, 'global_k': 64, 'quantization': 'fp16'},
                    {'name': 'opt_indexer_localglobal_fixed25', 'indexer_heads': 2, 'indexer_dim': 32, 'pattern': 'local_global', 'local_window': 32, 'global_k': 64, 'selector': 'fixed_ratio', 'ratio': 0.25},
                    {'name': 'opt_indexer_fp16_fixed25', 'indexer_heads': 2, 'indexer_dim': 32, 'quantization': 'fp16', 'selector': 'fixed_ratio', 'ratio': 0.25},
                    {'name': 'localglobal_fp16_fixed25', 'pattern': 'local_global', 'local_window': 32, 'global_k': 64, 'quantization': 'fp16', 'selector': 'fixed_ratio', 'ratio': 0.25},
                    
                    # Quadruple combination
                    {'name': 'all_optimizations', 'indexer_heads': 2, 'indexer_dim': 32, 'pattern': 'local_global', 'local_window': 32, 'global_k': 64, 'quantization': 'fp16', 'selector': 'fixed_ratio', 'ratio': 0.25},
                ],
                'sequence_lengths': [128, 512, 1024, 2048],
                'metrics': ['loss', 'accuracy', 'training_time', 'memory_usage', 'parameter_count', 'efficiency_score']
            },
        }
        
        return research_questions
    
    def run_research_question(self, question_name: str, question_config: Dict[str, Any]):
        """Run ablation study for a specific research question"""
        
        print(f"\n{'='*100}")
        print(f"RESEARCH QUESTION: {question_config['question']}")
        print(f"HYPOTHESIS: {question_config['hypothesis']}")
        print(f"{'='*100}")
        
        question_results = []
        
        for ablation_config in question_config['ablations']:
            ablation_name = ablation_config['name']
            
            for seq_len in question_config['sequence_lengths']:
                print(f"\nRunning {ablation_name} with sequence length {seq_len}")
                
                result = self.run_single_ablation(
                    f"{question_name}_{ablation_name}_seq{seq_len}",
                    ablation_config,
                    seq_len
                )
                
                question_results.append(result)
        
        # Save question results
        with open(f'ablation_results/{question_name}_results.json', 'w') as f:
            json.dump(question_results, f, indent=2)
        
        # Analyze results for this question
        self.analyze_research_question(question_name, question_config, question_results)
        
        return question_results
    
    def analyze_research_question(self, question_name: str, question_config: Dict[str, Any], results: List[Dict]):
        """Analyze results for a specific research question"""
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR: {question_config['question']}")
        print(f"{'='*80}")
        
        # Group results by ablation and sequence length
        ablation_results = {}
        for result in results:
            if 'error' in result:
                continue
            
            ablation_name = result['ablation_name'].split('_')[-2]  # Extract ablation name
            seq_len = result['sequence_length']
            
            if ablation_name not in ablation_results:
                ablation_results[ablation_name] = {}
            ablation_results[ablation_name][seq_len] = result
        
        # Create analysis based on question type
        if question_name == 'optimal_indexer_config':
            self._analyze_indexer_config(ablation_results, question_config)
        elif question_name == 'attention_pattern_analysis':
            self._analyze_attention_patterns(ablation_results, question_config)
        elif question_name == 'k_value_optimization':
            self._analyze_k_values(ablation_results, question_config)
        elif question_name == 'quantization_effectiveness':
            self._analyze_quantization(ablation_results, question_config)
        elif question_name == 'optimization_combinations':
            self._analyze_combinations(ablation_results, question_config)
        
        print(f"\nAnalysis complete for {question_name}")
    
    def _analyze_indexer_config(self, results: Dict, config: Dict):
        """Analyze indexer configuration results"""
        print("\nIndexer Configuration Analysis:")
        
        # Find best configuration for each sequence length
        for seq_len in config['sequence_lengths']:
            best_loss = float('inf')
            best_config = None
            
            for ablation_name, ablation_data in results.items():
                if seq_len in ablation_data:
                    loss = ablation_data[seq_len]['final_loss']
                    if loss < best_loss:
                        best_loss = loss
                        best_config = ablation_name
            
            print(f"  Seq Len {seq_len}: Best = {best_config} (loss = {best_loss:.4f})")
    
    def _analyze_attention_patterns(self, results: Dict, config: Dict):
        """Analyze attention pattern results"""
        print("\nAttention Pattern Analysis:")
        
        # Compare patterns across sequence lengths
        patterns = {}
        for ablation_name, ablation_data in results.items():
            pattern_type = ablation_name.split('_')[0]  # Extract pattern type
            if pattern_type not in patterns:
                patterns[pattern_type] = []
            
            for seq_len, result in ablation_data.items():
                patterns[pattern_type].append({
                    'seq_len': seq_len,
                    'loss': result['final_loss'],
                    'time': result['avg_training_time']
                })
        
        # Find best pattern for each sequence length
        for seq_len in config['sequence_lengths']:
            best_pattern = None
            best_score = float('inf')
            
            for pattern_type, pattern_results in patterns.items():
                pattern_result = next((r for r in pattern_results if r['seq_len'] == seq_len), None)
                if pattern_result:
                    # Combine loss and time into efficiency score
                    efficiency_score = pattern_result['loss'] * pattern_result['time']
                    if efficiency_score < best_score:
                        best_score = efficiency_score
                        best_pattern = pattern_type
            
            print(f"  Seq Len {seq_len}: Best Pattern = {best_pattern}")
    
    def _analyze_k_values(self, results: Dict, config: Dict):
        """Analyze k-value optimization results"""
        print("\nK-Value Optimization Analysis:")
        
        # Group by strategy type
        strategies = {'fixed': [], 'progressive': [], 'adaptive': []}
        
        for ablation_name, ablation_data in results.items():
            strategy_type = ablation_name.split('_')[0]
            if strategy_type in strategies:
                for seq_len, result in ablation_data.items():
                    strategies[strategy_type].append({
                        'seq_len': seq_len,
                        'loss': result['final_loss']
                    })
        
        # Find best strategy for each sequence length
        for seq_len in config['sequence_lengths']:
            best_strategy = None
            best_loss = float('inf')
            
            for strategy_type, strategy_results in strategies.items():
                strategy_result = next((r for r in strategy_results if r['seq_len'] == seq_len), None)
                if strategy_result and strategy_result['loss'] < best_loss:
                    best_loss = strategy_result['loss']
                    best_strategy = strategy_type
            
            print(f"  Seq Len {seq_len}: Best Strategy = {best_strategy}")
    
    def _analyze_quantization(self, results: Dict, config: Dict):
        """Analyze quantization effectiveness"""
        print("\nQuantization Analysis:")
        
        # Compare quantization types
        quant_types = {'fp32': [], 'fp16': [], 'int8': [], 'int4': []}
        
        for ablation_name, ablation_data in results.items():
            if 'fp32' in ablation_name:
                quant_type = 'fp32'
            elif 'fp16' in ablation_name:
                quant_type = 'fp16'
            elif 'int8' in ablation_name:
                quant_type = 'int8'
            elif 'int4' in ablation_name:
                quant_type = 'int4'
            else:
                continue
            
            for seq_len, result in ablation_data.items():
                quant_types[quant_type].append({
                    'seq_len': seq_len,
                    'loss': result['final_loss'],
                    'time': result['avg_training_time'],
                    'memory': result['memory_allocated_mb']
                })
        
        # Calculate speedup and quality impact
        for quant_type, quant_results in quant_types.items():
            if not quant_results:
                continue
                
            avg_loss = np.mean([r['loss'] for r in quant_results])
            avg_time = np.mean([r['time'] for r in quant_results])
            avg_memory = np.mean([r['memory'] for r in quant_results])
            
            print(f"  {quant_type.upper()}: Loss = {avg_loss:.4f}, Time = {avg_time:.3f}s, Memory = {avg_memory:.1f}MB")
    
    def _analyze_combinations(self, results: Dict, config: Dict):
        """Analyze optimization combinations"""
        print("\nOptimization Combination Analysis:")
        
        # Find best combination for each sequence length
        for seq_len in config['sequence_lengths']:
            best_combination = None
            best_efficiency = float('inf')
            
            for combination_name, combination_data in results.items():
                if seq_len in combination_data:
                    result = combination_data[seq_len]
                    # Calculate efficiency score (lower is better)
                    efficiency = result['final_loss'] * result['avg_training_time'] / result['total_params'] * 1e6
                    if efficiency < best_efficiency:
                        best_efficiency = efficiency
                        best_combination = combination_name
            
            print(f"  Seq Len {seq_len}: Best Combination = {best_combination}")
    
    def run_all_research_questions(self):
        """Run all research question ablation studies"""
        
        research_questions = self.define_research_questions()
        
        print("Starting Specialized Research Question Ablations")
        print(f"Research Questions: {list(research_questions.keys())}")
        
        all_results = {}
        
        for question_name, question_config in research_questions.items():
            question_results = self.run_research_question(question_name, question_config)
            all_results[question_name] = question_results
        
        # Save comprehensive results
        with open('ablation_results/specialized_research_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*100}")
        print("SPECIALIZED RESEARCH QUESTION ABLATIONS COMPLETE!")
        print(f"{'='*100}")
        print("Results saved to ablation_results/specialized_research_results.json")


def main():
    """Main function for specialized ablation studies"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Specialized Ablation Studies')
    parser.add_argument('--questions', nargs='+', 
                       choices=['optimal_indexer_config', 'attention_pattern_analysis', 'k_value_optimization', 'quantization_effectiveness', 'optimization_combinations'],
                       default=['optimal_indexer_config', 'attention_pattern_analysis'],
                       help='Research questions to investigate')
    parser.add_argument('--steps', type=int, default=500, help='Training steps per ablation')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = AblationConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    # Run specialized ablations
    runner = SpecializedAblationRunner(config)
    
    if len(args.questions) == 1 and args.questions[0] == 'all':
        runner.run_all_research_questions()
    else:
        research_questions = runner.define_research_questions()
        for question in args.questions:
            if question in research_questions:
                runner.run_research_question(question, research_questions[question])


if __name__ == '__main__':
    main()
