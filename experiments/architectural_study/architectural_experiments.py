"""
Comprehensive architectural experiments to find improvements over baseline.

This script conducts systematic experiments testing different architectural
choices while maintaining fair comparison conditions.
"""

import torch
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
from configs import AdaptiveMoEModelConfig
from models import create_model
from data import load_and_cache_data
from data.dataset import create_dataset
from training.trainer import train_model
from torch.utils.data import DataLoader
import numpy as np


class ExperimentConfig:
    """Configuration for a single experiment."""
    
    def __init__(self, name: str, config: AdaptiveMoEModelConfig, description: str):
        self.name = name
        self.config = config
        self.description = description


class ArchitecturalExperimentRunner:
    """Runs systematic architectural experiments."""
    
    def __init__(self):
        self.results = []
        self.baseline_config = None
        
    def get_baseline_config(self) -> AdaptiveMoEModelConfig:
        """Get the baseline configuration (our previous winner)."""
        return AdaptiveMoEModelConfig(
            # Model architecture - baseline (490 Max adapted)
            d_model=512,
            n_heads=16,
            n_layers=8,
            d_ff=2048,
            
            # Training parameters
            batch_size=16,
            max_steps=300,  # Shorter for more experiments
            gradient_accumulation_steps=2,
            
            # MoE configuration
            num_experts=16,
            expert_top_k=2,
            load_balancing_weight=0.01,
            
            # Data parameters
            max_seq_len=256,
            num_documents=1000,
            max_tokens=100000,
            
            # Training optimization
            muon_lr=0.01,
            use_amp=True,
            use_fp8=False,
            
            # Evaluation
            eval_every=100,
            eval_steps=25,
            
            # Regularization
            weight_decay=0.1,
            dropout=0.1,
            grad_clip=1.0,
            
            # GPU optimization
            use_adaptive_matmul=True,
            use_megatron=False,
        )
    
    def create_config_variant(self, baseline: AdaptiveMoEModelConfig, **overrides) -> AdaptiveMoEModelConfig:
        """Create a configuration variant with overrides."""
        config_dict = {
            'd_model': baseline.d_model,
            'n_heads': baseline.n_heads,
            'n_layers': baseline.n_layers,
            'd_ff': baseline.d_ff,
            'batch_size': baseline.batch_size,
            'max_steps': baseline.max_steps,
            'gradient_accumulation_steps': baseline.gradient_accumulation_steps,
            'num_experts': baseline.num_experts,
            'expert_top_k': baseline.expert_top_k,
            'load_balancing_weight': baseline.load_balancing_weight,
            'max_seq_len': baseline.max_seq_len,
            'num_documents': baseline.num_documents,
            'max_tokens': baseline.max_tokens,
            'muon_lr': baseline.muon_lr,
            'use_amp': baseline.use_amp,
            'use_fp8': baseline.use_fp8,
            'eval_every': baseline.eval_every,
            'eval_steps': baseline.eval_steps,
            'weight_decay': baseline.weight_decay,
            'dropout': baseline.dropout,
            'grad_clip': baseline.grad_clip,
            'use_adaptive_matmul': baseline.use_adaptive_matmul,
            'use_megatron': baseline.use_megatron,
        }
        config_dict.update(overrides)
        return AdaptiveMoEModelConfig(**config_dict)

    def get_experimental_configs(self) -> List[ExperimentConfig]:
        """Get all experimental configurations to test."""
        configs = []
        
        # Baseline
        baseline = self.get_baseline_config()
        configs.append(ExperimentConfig(
            "Baseline", baseline, "Original 490 Max configuration (adapted to 512)"
        ))
        
        # Experiment 1: Deeper Model
        deeper_config = self.create_config_variant(baseline, 
            d_model=448, n_heads=14, n_layers=12, d_ff=1792)
        configs.append(ExperimentConfig(
            "Deeper", deeper_config, "12 layers instead of 8, adjusted dimensions"
        ))
        
        # Experiment 2: Wider Model
        wider_config = self.create_config_variant(baseline,
            d_model=640, n_heads=20, d_ff=2560)
        configs.append(ExperimentConfig(
            "Wider", wider_config, "640 dimensions instead of 512, more attention heads"
        ))
        
        # Experiment 3: More Experts
        more_experts_config = self.create_config_variant(baseline,
            num_experts=24)
        configs.append(ExperimentConfig(
            "MoreExperts", more_experts_config, "24 experts instead of 16"
        ))
        
        # Experiment 4: Top-3 Experts
        top3_config = self.create_config_variant(baseline,
            expert_top_k=3)
        configs.append(ExperimentConfig(
            "Top3Experts", top3_config, "Top-3 expert selection instead of top-2"
        ))
        
        # Experiment 5: Lower Dropout
        low_dropout_config = self.create_config_variant(baseline,
            dropout=0.05)
        configs.append(ExperimentConfig(
            "LowDropout", low_dropout_config, "0.05 dropout instead of 0.1"
        ))
        
        # Experiment 6: Higher Learning Rate
        high_lr_config = self.create_config_variant(baseline,
            muon_lr=0.015)
        configs.append(ExperimentConfig(
            "HighLR", high_lr_config, "0.015 learning rate instead of 0.01"
        ))
        
        # Experiment 7: Smaller Batch Size
        small_batch_config = self.create_config_variant(baseline,
            batch_size=8, gradient_accumulation_steps=4)
        configs.append(ExperimentConfig(
            "SmallBatch", small_batch_config, "Batch size 8 with more accumulation"
        ))
        
        # Experiment 8: Longer Sequences
        long_seq_config = self.create_config_variant(baseline,
            max_seq_len=384)
        configs.append(ExperimentConfig(
            "LongSeq", long_seq_config, "384 sequence length instead of 256"
        ))
        
        return configs
    
    def run_single_experiment(self, exp_config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment."""
        print(f"\n🚀 Starting {exp_config.name} experiment...")
        print(f"📝 {exp_config.description}")
        print("=" * 60)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Load data
        print("📊 Loading data...")
        texts, tokenizer, tokens = load_and_cache_data(exp_config.config)
        exp_config.config.vocab_size = len(tokenizer)
        
        # Create datasets by splitting tokens
        split_idx = int(len(tokens) * 0.9)
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        
        train_dataset = create_dataset(train_tokens, dataset_type="text_token", seq_len=exp_config.config.max_seq_len)
        val_dataset = create_dataset(val_tokens, dataset_type="text_token", seq_len=exp_config.config.max_seq_len)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=exp_config.config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=exp_config.config.batch_size, shuffle=False, num_workers=0)
        
        # Create model
        print("🏗️ Creating model...")
        model = create_model(exp_config.config, model_type="moe")
        model.train()
        
        # Train model
        print("🎯 Starting training...")
        start_time = time.time()
        
        try:
            trained_model, metrics = train_model(model, train_loader, val_loader, exp_config.config)
            
            training_time = time.time() - start_time
            
            # Add experiment info to metrics
            metrics.update({
                'experiment_name': exp_config.name,
                'description': exp_config.description,
                'training_time_minutes': training_time / 60,
                'config_summary': {
                    'd_model': exp_config.config.d_model,
                    'n_layers': exp_config.config.n_layers,
                    'n_heads': exp_config.config.n_heads,
                    'num_experts': exp_config.config.num_experts,
                    'expert_top_k': exp_config.config.expert_top_k,
                    'batch_size': exp_config.config.batch_size,
                    'muon_lr': exp_config.config.muon_lr,
                    'dropout': exp_config.config.dropout,
                    'max_seq_len': exp_config.config.max_seq_len,
                }
            })
            
            print(f"\n✅ {exp_config.name} completed successfully!")
            print(f"📊 Final metrics:")
            print(f"   Val Loss: {metrics['val_loss']:.4f}")
            print(f"   Val Accuracy: {metrics['val_accuracy']:.4f}")
            print(f"   Val Perplexity: {metrics['val_perplexity']:.2f}")
            print(f"   Training Time: {metrics['training_time_minutes']:.1f} minutes")
            
            return metrics
            
        except Exception as e:
            print(f"❌ {exp_config.name} failed: {e}")
            return {
                'experiment_name': exp_config.name,
                'description': exp_config.description,
                'error': str(e),
                'val_loss': float('inf'),
                'val_accuracy': 0.0,
                'val_perplexity': float('inf'),
                'training_time_minutes': 0.0
            }
    
    def run_all_experiments(self):
        """Run all experiments."""
        print("🧪 Starting Comprehensive Architectural Experiments")
        print("=" * 70)
        
        configs = self.get_experimental_configs()
        
        for exp_config in configs:
            result = self.run_single_experiment(exp_config)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
        
        print("\n🏆 ALL EXPERIMENTS COMPLETED!")
        self.analyze_results()
    
    def save_results(self):
        """Save results to JSON file."""
        with open('/root/blueberry-llm/experiment_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def analyze_results(self):
        """Analyze and visualize results."""
        print("\n📊 EXPERIMENT ANALYSIS")
        print("=" * 70)
        
        # Create DataFrame for analysis
        df_data = []
        for result in self.results:
            if 'error' not in result:
                df_data.append({
                    'Experiment': result['experiment_name'],
                    'Val_Loss': result['val_loss'],
                    'Val_Accuracy': result['val_accuracy'],
                    'Val_Perplexity': result['val_perplexity'],
                    'Training_Time': result['training_time_minutes'],
                    'Best_Val_Loss': result.get('best_val_loss', result['val_loss']),
                    'd_model': result['config_summary']['d_model'],
                    'n_layers': result['config_summary']['n_layers'],
                    'n_heads': result['config_summary']['n_heads'],
                    'num_experts': result['config_summary']['num_experts'],
                    'expert_top_k': result['config_summary']['expert_top_k'],
                    'batch_size': result['config_summary']['batch_size'],
                    'muon_lr': result['config_summary']['muon_lr'],
                    'dropout': result['config_summary']['dropout'],
                    'max_seq_len': result['config_summary']['max_seq_len'],
                })
        
        df = pd.DataFrame(df_data)
        
        # Sort by validation loss (lower is better)
        df_sorted = df.sort_values('Val_Loss')
        
        print("\n🏆 RANKING BY VALIDATION LOSS (Lower is Better):")
        print("-" * 70)
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"{i:2d}. {row['Experiment']:<15} | Loss: {row['Val_Loss']:.4f} | Acc: {row['Val_Accuracy']:.4f} | Time: {row['Training_Time']:.1f}min")
        
        # Find best performing experiment
        best_exp = df_sorted.iloc[0]
        print(f"\n🥇 BEST EXPERIMENT: {best_exp['Experiment']}")
        print(f"   Validation Loss: {best_exp['Val_Loss']:.4f}")
        print(f"   Validation Accuracy: {best_exp['Val_Accuracy']:.4f}")
        print(f"   Training Time: {best_exp['Training_Time']:.1f} minutes")
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Save detailed analysis
        self.save_detailed_analysis(df_sorted)
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualizations of the results."""
        print("\n📈 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Architectural Experiment Results', fontsize=16, fontweight='bold')
        
        # 1. Validation Loss Comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(df['Experiment'], df['Val_Loss'], color='skyblue', alpha=0.7)
        ax1.set_title('Validation Loss by Experiment', fontweight='bold')
        ax1.set_ylabel('Validation Loss')
        ax1.tick_params(axis='x', rotation=45)
        
        # Highlight best result
        best_idx = df['Val_Loss'].idxmin()
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(1.0)
        
        # 2. Validation Accuracy Comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(df['Experiment'], df['Val_Accuracy'], color='lightgreen', alpha=0.7)
        ax2.set_title('Validation Accuracy by Experiment', fontweight='bold')
        ax2.set_ylabel('Validation Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        # Highlight best result
        best_acc_idx = df['Val_Accuracy'].idxmax()
        bars[best_acc_idx].set_color('gold')
        bars[best_acc_idx].set_alpha(1.0)
        
        # 3. Training Time Comparison
        ax3 = axes[0, 2]
        bars = ax3.bar(df['Experiment'], df['Training_Time'], color='lightcoral', alpha=0.7)
        ax3.set_title('Training Time by Experiment', fontweight='bold')
        ax3.set_ylabel('Training Time (minutes)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Highlight fastest result
        fastest_idx = df['Training_Time'].idxmin()
        bars[fastest_idx].set_color('gold')
        bars[fastest_idx].set_alpha(1.0)
        
        # 4. Model Size vs Performance Scatter
        ax4 = axes[1, 0]
        scatter = ax4.scatter(df['d_model'] * df['n_layers'], df['Val_Loss'], 
                            s=100, alpha=0.7, c=df['num_experts'], cmap='viridis')
        ax4.set_title('Model Size vs Validation Loss', fontweight='bold')
        ax4.set_xlabel('Model Size (d_model × n_layers)')
        ax4.set_ylabel('Validation Loss')
        plt.colorbar(scatter, ax=ax4, label='Number of Experts')
        
        # 5. Learning Rate vs Performance
        ax5 = axes[1, 1]
        ax5.scatter(df['muon_lr'], df['Val_Loss'], s=100, alpha=0.7, color='purple')
        ax5.set_title('Learning Rate vs Validation Loss', fontweight='bold')
        ax5.set_xlabel('Learning Rate')
        ax5.set_ylabel('Validation Loss')
        
        # 6. Dropout vs Performance
        ax6 = axes[1, 2]
        ax6.scatter(df['dropout'], df['Val_Loss'], s=100, alpha=0.7, color='orange')
        ax6.set_title('Dropout vs Validation Loss', fontweight='bold')
        ax6.set_xlabel('Dropout Rate')
        ax6.set_ylabel('Validation Loss')
        
        plt.tight_layout()
        plt.savefig('/root/blueberry-llm/experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create correlation heatmap
        self.create_correlation_heatmap(df)
        
        print("✅ Visualizations saved to experiment_results.png")
    
    def create_correlation_heatmap(self, df: pd.DataFrame):
        """Create correlation heatmap of architectural parameters."""
        # Select numerical columns for correlation
        numerical_cols = ['Val_Loss', 'Val_Accuracy', 'Val_Perplexity', 'Training_Time',
                         'd_model', 'n_layers', 'n_heads', 'num_experts', 'expert_top_k',
                         'batch_size', 'muon_lr', 'dropout', 'max_seq_len']
        
        corr_df = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(corr_df, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Architectural Parameters', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/root/blueberry-llm/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Correlation heatmap saved to correlation_heatmap.png")
    
    def save_detailed_analysis(self, df_sorted: pd.DataFrame):
        """Save detailed analysis to file."""
        with open('/root/blueberry-llm/detailed_analysis.txt', 'w') as f:
            f.write("ARCHITECTURAL EXPERIMENT ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("EXPERIMENT RANKINGS (by Validation Loss):\n")
            f.write("-" * 50 + "\n")
            for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
                f.write(f"{i:2d}. {row['Experiment']:<15} | Loss: {row['Val_Loss']:.4f} | Acc: {row['Val_Accuracy']:.4f} | Time: {row['Training_Time']:.1f}min\n")
            
            f.write(f"\nBEST EXPERIMENT: {df_sorted.iloc[0]['Experiment']}\n")
            f.write(f"Validation Loss: {df_sorted.iloc[0]['Val_Loss']:.4f}\n")
            f.write(f"Validation Accuracy: {df_sorted.iloc[0]['Val_Accuracy']:.4f}\n")
            f.write(f"Training Time: {df_sorted.iloc[0]['Training_Time']:.1f} minutes\n")
            
            f.write("\nKEY INSIGHTS:\n")
            f.write("-" * 50 + "\n")
            
            # Find insights
            best_loss = df_sorted.iloc[0]
            worst_loss = df_sorted.iloc[-1]
            
            f.write(f"• Best performing experiment: {best_loss['Experiment']}\n")
            f.write(f"• Worst performing experiment: {worst_loss['Experiment']}\n")
            f.write(f"• Performance improvement: {worst_loss['Val_Loss'] - best_loss['Val_Loss']:.4f} loss reduction\n")
            
            # Find fastest and slowest
            fastest = df_sorted.loc[df_sorted['Training_Time'].idxmin()]
            slowest = df_sorted.loc[df_sorted['Training_Time'].idxmax()]
            
            f.write(f"• Fastest training: {fastest['Experiment']} ({fastest['Training_Time']:.1f} min)\n")
            f.write(f"• Slowest training: {slowest['Experiment']} ({slowest['Training_Time']:.1f} min)\n")
        
        print("✅ Detailed analysis saved to detailed_analysis.txt")


def main():
    """Main function to run all experiments."""
    runner = ArchitecturalExperimentRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
