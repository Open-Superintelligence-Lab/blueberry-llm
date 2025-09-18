#!/usr/bin/env python3
"""
Analyze and visualize AMP experiment results
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def load_results(filename: str) -> List[Dict]:
    """Load experiment results from JSON file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'r') as f:
        return json.load(f)

def create_performance_comparison(results: List[Dict]):
    """Create performance comparison charts"""
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Filter successful experiments
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("❌ No successful experiments to analyze")
        return
    
    # Create model size labels
    df_success['model_size'] = df_success['config'].apply(
        lambda x: f"{x['d_model']}d×{x['n_layers']}L"
    )
    df_success['batch_size'] = df_success['config'].apply(lambda x: x['batch_size'])
    df_success['use_amp'] = df_success['config'].apply(lambda x: x['use_amp'])
    df_success['precision'] = df_success['use_amp'].map({True: 'AMP', False: 'FP32'})
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AMP vs FP32 Performance Analysis', fontsize=16)
    
    # 1. Speed comparison by model size
    ax1 = axes[0, 0]
    speed_data = df_success.groupby(['model_size', 'precision'])['tokens_per_second'].mean().unstack()
    speed_data.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e'])
    ax1.set_title('Training Speed by Model Size')
    ax1.set_ylabel('Tokens/Second')
    ax1.legend(title='Precision')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Memory usage comparison
    ax2 = axes[0, 1]
    memory_data = df_success.groupby(['model_size', 'precision'])['peak_memory_mb'].mean().unstack()
    memory_data.plot(kind='bar', ax=ax2, color=['#1f77b4', '#ff7f0e'])
    ax2.set_title('Memory Usage by Model Size')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.legend(title='Precision')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Speed vs Batch Size
    ax3 = axes[1, 0]
    for precision in ['AMP', 'FP32']:
        data = df_success[df_success['precision'] == precision]
        ax3.plot(data['batch_size'], data['tokens_per_second'], 
                'o-', label=precision, linewidth=2, markersize=6)
    ax3.set_title('Speed vs Batch Size')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Tokens/Second')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Memory vs Batch Size
    ax4 = axes[1, 1]
    for precision in ['AMP', 'FP32']:
        data = df_success[df_success['precision'] == precision]
        ax4.plot(data['batch_size'], data['peak_memory_mb'], 
                'o-', label=precision, linewidth=2, markersize=6)
    ax4.set_title('Memory vs Batch Size')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Peak Memory (MB)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), 'performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance analysis saved to: {plot_path}")
    
    return plot_path

def create_amp_benefit_analysis(results: List[Dict]):
    """Create analysis showing when AMP is beneficial"""
    df = pd.DataFrame(results)
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("❌ No successful experiments to analyze")
        return
    
    # Prepare data
    df_success['model_size'] = df_success['config'].apply(
        lambda x: f"{x['d_model']}d×{x['n_layers']}L"
    )
    df_success['batch_size'] = df_success['config'].apply(lambda x: x['batch_size'])
    df_success['use_amp'] = df_success['config'].apply(lambda x: x['use_amp'])
    
    # Calculate AMP benefit (speed improvement)
    benefit_data = []
    for model_size in df_success['model_size'].unique():
        for batch_size in df_success['batch_size'].unique():
            fp32_data = df_success[
                (df_success['model_size'] == model_size) & 
                (df_success['batch_size'] == batch_size) & 
                (df_success['use_amp'] == False)
            ]
            amp_data = df_success[
                (df_success['model_size'] == model_size) & 
                (df_success['batch_size'] == batch_size) & 
                (df_success['use_amp'] == True)
            ]
            
            if len(fp32_data) > 0 and len(amp_data) > 0:
                fp32_speed = fp32_data['tokens_per_second'].iloc[0]
                amp_speed = amp_data['tokens_per_second'].iloc[0]
                speed_improvement = (amp_speed - fp32_speed) / fp32_speed * 100
                
                fp32_memory = fp32_data['peak_memory_mb'].iloc[0]
                amp_memory = amp_data['peak_memory_mb'].iloc[0]
                memory_savings = (fp32_memory - amp_memory) / fp32_memory * 100
                
                benefit_data.append({
                    'model_size': model_size,
                    'batch_size': batch_size,
                    'speed_improvement': speed_improvement,
                    'memory_savings': memory_savings,
                    'amp_better': speed_improvement > 0
                })
    
    benefit_df = pd.DataFrame(benefit_data)
    
    if len(benefit_df) == 0:
        print("❌ No comparison data available")
        return
    
    # Create benefit analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('AMP Benefit Analysis', fontsize=16)
    
    # Speed improvement heatmap
    ax1 = axes[0]
    pivot_speed = benefit_df.pivot(index='model_size', columns='batch_size', values='speed_improvement')
    sns.heatmap(pivot_speed, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax1)
    ax1.set_title('Speed Improvement (%)')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Model Size')
    
    # Memory savings heatmap
    ax2 = axes[1]
    pivot_memory = benefit_df.pivot(index='model_size', columns='batch_size', values='memory_savings')
    sns.heatmap(pivot_memory, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax2)
    ax2.set_title('Memory Savings (%)')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Model Size')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), 'amp_benefit_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 AMP benefit analysis saved to: {plot_path}")
    
    # Print summary
    print(f"\n🎯 AMP BENEFIT SUMMARY:")
    print(f"   Experiments where AMP is faster: {sum(benefit_df['amp_better'])}/{len(benefit_df)}")
    
    best_amp = benefit_df.loc[benefit_df['speed_improvement'].idxmax()]
    print(f"   Best AMP improvement: {best_amp['speed_improvement']:.1f}% speedup")
    print(f"   Configuration: {best_amp['model_size']}, batch={best_amp['batch_size']}")
    
    return plot_path

def print_detailed_results(results: List[Dict]):
    """Print detailed results table"""
    df = pd.DataFrame(results)
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("❌ No successful experiments to display")
        return
    
    # Prepare display data
    df_success['model_size'] = df_success['config'].apply(
        lambda x: f"{x['d_model']}d×{x['n_layers']}L"
    )
    df_success['batch_size'] = df_success['config'].apply(lambda x: x['batch_size'])
    df_success['precision'] = df_success['config'].apply(lambda x: 'AMP' if x['use_amp'] else 'FP32')
    
    # Select columns for display
    display_cols = ['model_size', 'batch_size', 'precision', 'tokens_per_second', 'peak_memory_mb', 'final_val_loss']
    display_df = df_success[display_cols].copy()
    
    # Round numerical columns
    display_df['tokens_per_second'] = display_df['tokens_per_second'].round(0)
    display_df['peak_memory_mb'] = display_df['peak_memory_mb'].round(0)
    display_df['final_val_loss'] = display_df['final_val_loss'].round(4)
    
    print("\n📊 DETAILED RESULTS:")
    print("=" * 80)
    print(display_df.to_string(index=False))
    
    return display_df

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze AMP experiment results")
    parser.add_argument("--file", type=str, help="Results file to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all result files")
    
    args = parser.parse_args()
    
    if args.all:
        # Find all result files
        result_files = [f for f in os.listdir(os.path.dirname(__file__)) 
                       if f.startswith('amp_experiment_results') and f.endswith('.json')]
        
        if not result_files:
            print("❌ No result files found")
            return
        
        print(f"📁 Found {len(result_files)} result files")
        for file in result_files:
            print(f"   Analyzing: {file}")
            results = load_results(file)
            print_detailed_results(results)
            create_performance_comparison(results)
            create_amp_benefit_analysis(results)
            print()
    
    elif args.file:
        results = load_results(args.file)
        print_detailed_results(results)
        create_performance_comparison(results)
        create_amp_benefit_analysis(results)
    
    else:
        print("❌ Please specify --file <filename> or --all")
        print("Available files:")
        result_files = [f for f in os.listdir(os.path.dirname(__file__)) 
                       if f.startswith('amp_experiment_results') and f.endswith('.json')]
        for file in result_files:
            print(f"   {file}")

if __name__ == "__main__":
    main()
