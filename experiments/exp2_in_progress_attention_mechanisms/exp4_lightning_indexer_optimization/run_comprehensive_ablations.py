#!/usr/bin/env python3
"""
Comprehensive Ablation Runner for Experiment 4

This script runs all comprehensive ablation studies for Lightning Indexer optimization.
It provides multiple levels of analysis from basic to advanced research questions.

Usage:
    # Run all comprehensive ablations
    python run_comprehensive_ablations.py --mode all

    # Run specific ablation categories
    python run_comprehensive_ablations.py --mode comprehensive --categories indexer_architecture attention_patterns

    # Run specific research questions
    python run_comprehensive_ablations.py --mode research --questions optimal_indexer_config k_value_optimization

    # Run quick test with limited configurations
    python run_comprehensive_ablations.py --mode quick --steps 100 --seq-lens 64 128
"""

import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from comprehensive_ablations import ComprehensiveAblationRunner, AblationConfig
from specialized_ablations import SpecializedAblationRunner


def create_ablation_config(args) -> AblationConfig:
    """Create ablation configuration from command line arguments"""
    return AblationConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_every=args.eval_every,
        sequence_lengths=args.seq_lens,
        seed=args.seed
    )


def run_quick_test(args):
    """Run quick test with limited configurations"""
    print("🚀 Running Quick Test Ablations")
    print("=" * 60)
    
    config = create_ablation_config(args)
    
    # Limit to basic configurations for quick test
    runner = ComprehensiveAblationRunner(config)
    
    # Run only essential ablations
    quick_categories = ['indexer_architecture']
    quick_ablations = {
        'indexer_architecture': [
            {'name': 'baseline_4h_64d', 'indexer_heads': 4, 'indexer_dim': 64},
            {'name': 'optimized_2h_32d', 'indexer_heads': 2, 'indexer_dim': 32},
            {'name': 'minimal_1h_16d', 'indexer_heads': 1, 'indexer_dim': 16},
            {'name': 'ultra_light_1h_8d', 'indexer_heads': 1, 'indexer_dim': 8},
        ]
    }
    
    # Override ablation definitions for quick test
    runner.define_ablation_studies = lambda: quick_ablations
    
    runner.run_comprehensive_ablations(quick_categories)
    
    print("✅ Quick test complete!")


def run_comprehensive_ablations(args):
    """Run comprehensive ablation studies"""
    print("🔬 Running Comprehensive Ablation Studies")
    print("=" * 60)
    
    config = create_ablation_config(args)
    runner = ComprehensiveAblationRunner(config)
    
    categories = args.categories if args.categories else [
        'indexer_architecture',
        'attention_patterns', 
        'quantization',
        'adaptive_selection'
    ]
    
    runner.run_comprehensive_ablations(categories)
    
    print("✅ Comprehensive ablations complete!")


def run_research_questions(args):
    """Run specialized research question ablations"""
    print("🎯 Running Research Question Ablations")
    print("=" * 60)
    
    config = create_ablation_config(args)
    runner = SpecializedAblationRunner(config)
    
    questions = args.questions if args.questions else [
        'optimal_indexer_config',
        'attention_pattern_analysis',
        'k_value_optimization'
    ]
    
    research_questions = runner.define_research_questions()
    
    for question in questions:
        if question in research_questions:
            runner.run_research_question(question, research_questions[question])
        else:
            print(f"⚠️  Unknown research question: {question}")
    
    print("✅ Research question ablations complete!")


def run_full_analysis(args):
    """Run full comprehensive analysis"""
    print("🌟 Running Full Comprehensive Analysis")
    print("=" * 60)
    
    config = create_ablation_config(args)
    
    # Phase 1: Comprehensive ablations
    print("\n📊 Phase 1: Comprehensive Ablations")
    comprehensive_runner = ComprehensiveAblationRunner(config)
    comprehensive_runner.run_comprehensive_ablations([
        'indexer_architecture',
        'attention_patterns',
        'quantization', 
        'adaptive_selection',
        'combined_strategies'
    ])
    
    # Phase 2: Research questions
    print("\n🔬 Phase 2: Research Question Analysis")
    research_runner = SpecializedAblationRunner(config)
    research_questions = research_runner.define_research_questions()
    
    for question_name, question_config in research_questions.items():
        research_runner.run_research_question(question_name, question_config)
    
    # Phase 3: Create final comprehensive analysis
    print("\n📈 Phase 3: Final Comprehensive Analysis")
    create_final_analysis()
    
    print("✅ Full comprehensive analysis complete!")


def create_final_analysis():
    """Create final comprehensive analysis combining all results"""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Load all results
        all_results = []
        
        # Load comprehensive results
        if os.path.exists('ablation_results/comprehensive_results.json'):
            with open('ablation_results/comprehensive_results.json', 'r') as f:
                comprehensive_results = json.load(f)
                all_results.extend(comprehensive_results)
        
        # Load specialized results
        if os.path.exists('ablation_results/specialized_research_results.json'):
            with open('ablation_results/specialized_research_results.json', 'r') as f:
                specialized_results = json.load(f)
                for question_results in specialized_results.values():
                    all_results.extend(question_results)
        
        if not all_results:
            print("⚠️  No results found for final analysis")
            return
        
        # Filter out errors
        valid_results = [r for r in all_results if 'error' not in r]
        
        if not valid_results:
            print("⚠️  No valid results found for final analysis")
            return
        
        # Create comprehensive analysis plots
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('Final Comprehensive Analysis: Lightning Indexer Optimization', fontsize=20)
        
        # Extract data for plotting
        losses = [r['final_loss'] for r in valid_results]
        times = [r['avg_training_time'] for r in valid_results]
        params = [r['indexer_params'] for r in valid_results]
        seq_lens = [r['sequence_length'] for r in valid_results]
        names = [r['ablation_name'] for r in valid_results]
        
        # Plot 1: Parameter vs Performance
        ax1 = axes[0, 0]
        ax1.scatter(params, losses, alpha=0.7, s=50)
        ax1.set_xlabel('Indexer Parameters')
        ax1.set_ylabel('Final Loss')
        ax1.set_title('Parameter Count vs Performance')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speed vs Quality
        ax2 = axes[0, 1]
        ax2.scatter(times, losses, alpha=0.7, s=50)
        ax2.set_xlabel('Training Time (s)')
        ax2.set_ylabel('Final Loss')
        ax2.set_title('Speed vs Quality')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sequence Length Scaling
        ax3 = axes[0, 2]
        unique_seq_lens = sorted(set(seq_lens))
        for seq_len in unique_seq_lens:
            seq_results = [r for r in valid_results if r['sequence_length'] == seq_len]
            seq_losses = [r['final_loss'] for r in seq_results]
            seq_times = [r['avg_training_time'] for r in seq_results]
            ax3.scatter(seq_times, seq_losses, label=f'Seq={seq_len}', alpha=0.7, s=50)
        ax3.set_xlabel('Training Time (s)')
        ax3.set_ylabel('Final Loss')
        ax3.set_title('Sequence Length Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Pareto Frontier
        ax4 = axes[1, 0]
        ax4.scatter(times, losses, alpha=0.7, s=50)
        ax4.set_xlabel('Training Time (s)')
        ax4.set_ylabel('Final Loss')
        ax4.set_title('Pareto Frontier')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Parameter Efficiency
        ax5 = axes[1, 1]
        param_ratios = [r['indexer_param_ratio'] for r in valid_results]
        ax5.scatter(param_ratios, losses, alpha=0.7, s=50)
        ax5.set_xlabel('Indexer Parameter Ratio')
        ax5.set_ylabel('Final Loss')
        ax5.set_title('Parameter Efficiency')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Best Configurations
        ax6 = axes[1, 2]
        best_configs = {}
        for seq_len in unique_seq_lens:
            seq_results = [r for r in valid_results if r['sequence_length'] == seq_len]
            if seq_results:
                best_result = min(seq_results, key=lambda x: x['final_loss'])
                best_configs[seq_len] = best_result
        
        best_seqs = list(best_configs.keys())
        best_losses = [best_configs[seq]['final_loss'] for seq in best_seqs]
        
        bars = ax6.bar([str(s) for s in best_seqs], best_losses, alpha=0.7)
        ax6.set_xlabel('Sequence Length')
        ax6.set_ylabel('Best Loss')
        ax6.set_title('Best Configuration by Sequence Length')
        
        # Add value labels
        for bar, value in zip(bars, best_losses):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(best_losses)*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 7: Improvement Analysis
        ax7 = axes[2, 0]
        baseline_results = [r for r in valid_results if 'baseline' in r['ablation_name']]
        if baseline_results:
            baseline_loss = min(r['final_loss'] for r in baseline_results)
            baseline_time = min(r['avg_training_time'] for r in baseline_results)
            
            improvements = []
            speed_improvements = []
            for result in valid_results:
                loss_improvement = (baseline_loss - result['final_loss']) / baseline_loss * 100
                speed_improvement = (baseline_time - result['avg_training_time']) / baseline_time * 100
                improvements.append(loss_improvement)
                speed_improvements.append(speed_improvement)
            
            ax7.scatter(speed_improvements, improvements, alpha=0.7, s=50)
            ax7.set_xlabel('Speed Improvement (%)')
            ax7.set_ylabel('Quality Improvement (%)')
            ax7.set_title('Improvements vs Baseline')
            ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax7.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Memory Usage
        ax8 = axes[2, 1]
        memory_usage = [r['memory_allocated_mb'] for r in valid_results]
        ax8.scatter(memory_usage, losses, alpha=0.7, s=50)
        ax8.set_xlabel('Memory Usage (MB)')
        ax8.set_ylabel('Final Loss')
        ax8.set_title('Memory Usage vs Performance')
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Summary Statistics
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Calculate summary statistics
        total_experiments = len(valid_results)
        avg_loss = np.mean(losses)
        avg_time = np.mean(times)
        min_loss = np.min(losses)
        min_time = np.min(times)
        
        summary_text = f"""
        Comprehensive Analysis Summary
        
        Total Experiments: {total_experiments}
        Average Loss: {avg_loss:.4f}
        Average Time: {avg_time:.3f}s
        Best Loss: {min_loss:.4f}
        Best Time: {min_time:.3f}s
        
        Sequence Lengths: {unique_seq_lens}
        Parameter Range: {min(params):,} - {max(params):,}
        """
        
        ax9.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('ablation_results/final_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary report
        create_summary_report(valid_results)
        
        print("✅ Final comprehensive analysis complete!")
        
    except Exception as e:
        print(f"❌ Error creating final analysis: {e}")
        import traceback
        traceback.print_exc()


def create_summary_report(results: List[Dict]):
    """Create comprehensive summary report"""
    
    # Group results by sequence length
    seq_groups = {}
    for result in results:
        seq_len = result['sequence_length']
        if seq_len not in seq_groups:
            seq_groups[seq_len] = []
        seq_groups[seq_len].append(result)
    
    # Find best configurations
    best_configs = {}
    for seq_len, seq_results in seq_groups.items():
        # Best by loss
        best_loss = min(seq_results, key=lambda x: x['final_loss'])
        # Best by time
        best_time = min(seq_results, key=lambda x: x['avg_training_time'])
        # Best by efficiency (loss * time / params)
        best_efficiency = min(seq_results, key=lambda x: x['final_loss'] * x['avg_training_time'] / x['indexer_params'])
        
        best_configs[seq_len] = {
            'best_loss': best_loss,
            'best_time': best_time,
            'best_efficiency': best_efficiency
        }
    
    # Create report
    report = {
        'summary': {
            'total_experiments': len(results),
            'sequence_lengths_tested': sorted(seq_groups.keys()),
            'best_configurations': best_configs
        },
        'key_findings': {
            'parameter_reduction': calculate_parameter_reduction(results),
            'speed_improvements': calculate_speed_improvements(results),
            'quality_impact': calculate_quality_impact(results)
        },
        'recommendations': generate_recommendations(best_configs)
    }
    
    # Save report
    with open('ablation_results/comprehensive_summary_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("📊 Comprehensive summary report saved to ablation_results/comprehensive_summary_report.json")


def calculate_parameter_reduction(results):
    """Calculate parameter reduction statistics"""
    baseline_params = max(r['indexer_params'] for r in results if 'baseline' in r['ablation_name'])
    reductions = []
    
    for result in results:
        if 'baseline' not in result['ablation_name']:
            reduction = (baseline_params - result['indexer_params']) / baseline_params * 100
            reductions.append(reduction)
    
    return {
        'baseline_parameters': baseline_params,
        'max_reduction': max(reductions) if reductions else 0,
        'avg_reduction': sum(reductions) / len(reductions) if reductions else 0,
        'reductions_over_50_percent': sum(1 for r in reductions if r > 50)
    }


def calculate_speed_improvements(results):
    """Calculate speed improvement statistics"""
    baseline_time = min(r['avg_training_time'] for r in results if 'baseline' in r['ablation_name'])
    improvements = []
    
    for result in results:
        if 'baseline' not in result['ablation_name']:
            improvement = (baseline_time - result['avg_training_time']) / baseline_time * 100
            improvements.append(improvement)
    
    return {
        'baseline_time': baseline_time,
        'max_improvement': max(improvements) if improvements else 0,
        'avg_improvement': sum(improvements) / len(improvements) if improvements else 0,
        'improvements_over_10_percent': sum(1 for i in improvements if i > 10)
    }


def calculate_quality_impact(results):
    """Calculate quality impact statistics"""
    baseline_loss = min(r['final_loss'] for r in results if 'baseline' in r['ablation_name'])
    impacts = []
    
    for result in results:
        if 'baseline' not in result['ablation_name']:
            impact = (result['final_loss'] - baseline_loss) / baseline_loss * 100
            impacts.append(impact)
    
    return {
        'baseline_loss': baseline_loss,
        'max_degradation': max(impacts) if impacts else 0,
        'avg_impact': sum(impacts) / len(impacts) if impacts else 0,
        'configurations_with_improved_quality': sum(1 for i in impacts if i < 0)
    }


def generate_recommendations(best_configs):
    """Generate recommendations based on best configurations"""
    recommendations = []
    
    for seq_len, configs in best_configs.items():
        best_loss = configs['best_loss']
        best_time = configs['best_time']
        best_efficiency = configs['best_efficiency']
        
        recommendations.append({
            'sequence_length': seq_len,
            'best_for_quality': best_loss['ablation_name'],
            'best_for_speed': best_time['ablation_name'],
            'best_for_efficiency': best_efficiency['ablation_name']
        })
    
    return recommendations


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Comprehensive Ablation Runner for Experiment 4')
    
    # Mode selection
    parser.add_argument('--mode', choices=['quick', 'comprehensive', 'research', 'all'], 
                       default='quick', help='Ablation mode to run')
    
    # Model configuration
    parser.add_argument('--vocab-size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--d-ff', type=int, default=512, help='FFN dimension')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Maximum sequence length')
    
    # Training configuration
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--steps', type=int, default=500, help='Training steps')
    parser.add_argument('--eval-every', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[64, 128, 256, 512], 
                       help='Sequence lengths to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Comprehensive mode options
    parser.add_argument('--categories', nargs='+', 
                       choices=['indexer_architecture', 'attention_patterns', 'quantization', 
                               'adaptive_selection', 'combined_strategies', 'scaling_analysis', 
                               'hardware_optimization'],
                       help='Ablation categories for comprehensive mode')
    
    # Research mode options
    parser.add_argument('--questions', nargs='+',
                       choices=['optimal_indexer_config', 'attention_pattern_analysis', 
                               'k_value_optimization', 'quantization_effectiveness', 
                               'optimization_combinations'],
                       help='Research questions for research mode')
    
    args = parser.parse_args()
    
    print("🔬 Lightning Indexer Optimization - Comprehensive Ablation Studies")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Steps: {args.steps}")
    print(f"Sequence Lengths: {args.seq_lens}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        if args.mode == 'quick':
            run_quick_test(args)
        elif args.mode == 'comprehensive':
            run_comprehensive_ablations(args)
        elif args.mode == 'research':
            run_research_questions(args)
        elif args.mode == 'all':
            run_full_analysis(args)
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 All ablations complete in {elapsed_time:.1f} seconds!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Ablations interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running ablations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
