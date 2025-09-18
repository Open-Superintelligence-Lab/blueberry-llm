#!/usr/bin/env python3
"""
AMP vs FP32 Experiment Runner for Google Colab
Tests when AMP becomes beneficial on T4 GPU
"""

import os
import sys
import time
import json
import torch
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # article/
project_root = os.path.dirname(parent_dir)  # blueberry-llm/
sys.path.insert(0, project_root)

try:
    from core.auto_config import AutoConfig
    from legacy.llm import train_moe_model, MoEModelConfig
    from torch.utils.data import DataLoader, random_split
    from data.loader import load_and_cache_data
    from data.dataset import TextTokenDataset
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print("Make sure you're running from the blueberry-llm project root directory")
    raise

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    # Model parameters (required)
    d_model: int
    n_layers: int
    batch_size: int
    
    # Model parameters (optional)
    n_heads: int = 8
    d_ff: int = None
    num_experts: int = 8
    
    # Training parameters
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.01
    max_seq_len: int = 512
    
    # Precision
    use_amp: bool = True
    
    # Data parameters
    num_documents: int = 1000
    max_tokens: int = 50000
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = self.d_model * 4

@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config: ExperimentConfig
    training_time: float
    tokens_per_second: float
    peak_memory_mb: float
    final_val_loss: float
    final_val_accuracy: float
    convergence_steps: int
    amp_scaling_stats: Optional[Dict] = None
    success: bool = True
    error_message: Optional[str] = None

class AMPExperimentRunner:
    """Main experiment runner for AMP vs FP32 testing"""
    
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.results: List[ExperimentResult] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Adjust parameters for test mode
        if test_mode:
            self.max_steps = 10
            self.num_documents = 100
            self.max_tokens = 5000
            print("🧪 TEST MODE: Running only 10 steps per experiment")
        else:
            self.max_steps = 1000
            self.num_documents = 1000
            self.max_tokens = 50000
            print("🚀 FULL MODE: Running 1000 steps per experiment")
    
    def generate_experiment_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations"""
        configs = []
        
        # Model sizes to test
        model_configs = [
            {"d_model": 128, "n_layers": 2, "n_heads": 4},   # Small
            {"d_model": 256, "n_layers": 4, "n_heads": 8},   # Medium  
            {"d_model": 384, "n_layers": 6, "n_heads": 8},   # Large (T4 default)
        ]
        
        # Batch sizes to test
        batch_sizes = [4, 8, 12]
        
        # Precision modes
        precision_modes = [False, True]  # FP32, AMP
        
        for model_config in model_configs:
            for batch_size in batch_sizes:
                for use_amp in precision_modes:
                    config = ExperimentConfig(
                        **model_config,
                        batch_size=batch_size,
                        max_steps=self.max_steps,
                        max_seq_len=512,
                        use_amp=use_amp,
                        num_documents=self.num_documents,
                        max_tokens=self.max_tokens,
                        gradient_accumulation_steps=max(1, 32 // batch_size)
                    )
                    configs.append(config)
        
        return configs
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment and return results"""
        print(f"\n🔬 Running experiment:")
        print(f"   Model: {config.d_model}d × {config.n_layers}L")
        print(f"   Batch: {config.batch_size}")
        print(f"   AMP: {'Yes' if config.use_amp else 'No'}")
        
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Record start time and memory
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Convert to MoEModelConfig
            model_config = MoEModelConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                batch_size=config.batch_size,
                max_steps=config.max_steps,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                muon_lr=config.learning_rate,
                max_seq_len=config.max_seq_len,
                num_experts=config.num_experts,
                use_amp=config.use_amp,
                num_documents=config.num_documents,
                max_tokens=config.max_tokens,
                vocab_size=50257  # GPT-2 vocab size
            )
            
            # Load data
            texts, tokenizer, tokens = load_and_cache_data(model_config)
            dataset = TextTokenDataset(tokens, model_config.max_seq_len)
            
            # Train/val split
            val_size = len(dataset) // 10
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=model_config.batch_size, 
                shuffle=True,
                num_workers=0  # Colab compatibility
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=model_config.batch_size, 
                shuffle=False,
                num_workers=0
            )
            
            # Train model
            model, final_metrics = train_moe_model(model_config, train_loader, val_loader)
            
            # Record end time and memory
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Calculate metrics
            training_time = end_time - start_time
            peak_memory_mb = (end_memory - start_memory) / (1024 * 1024)
            
            # Estimate tokens per second (rough calculation)
            total_tokens = len(train_dataset) * model_config.max_seq_len
            tokens_per_second = total_tokens / training_time if training_time > 0 else 0
            
            # Create result
            result = ExperimentResult(
                config=config,
                training_time=training_time,
                tokens_per_second=tokens_per_second,
                peak_memory_mb=peak_memory_mb,
                final_val_loss=final_metrics.get('val_loss', 0.0),
                final_val_accuracy=final_metrics.get('val_accuracy', 0.0),
                convergence_steps=model_config.max_steps,  # Simplified
                success=True
            )
            
            print(f"   ✅ Success: {training_time:.1f}s, {tokens_per_second:.0f} tok/s")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return ExperimentResult(
                config=config,
                training_time=0.0,
                tokens_per_second=0.0,
                peak_memory_mb=0.0,
                final_val_loss=0.0,
                final_val_accuracy=0.0,
                convergence_steps=0,
                success=False,
                error_message=str(e)
            )
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all experiments"""
        configs = self.generate_experiment_configs()
        total_experiments = len(configs)
        
        print(f"\n🚀 Starting {total_experiments} experiments")
        print(f"   Device: {self.device}")
        print(f"   Mode: {'TEST' if self.test_mode else 'FULL'}")
        
        for i, config in enumerate(configs, 1):
            print(f"\n📊 Progress: {i}/{total_experiments}")
            result = self.run_single_experiment(config)
            self.results.append(result)
            
            # Clear memory between experiments
            torch.cuda.empty_cache()
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            mode = "test" if self.test_mode else "full"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amp_experiment_results_{mode}_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            serializable_results.append(result_dict)
        
        # Save to file
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print experiment summary"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"\n📊 Experiment Summary:")
        print(f"   Total experiments: {len(self.results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        
        if successful:
            print(f"\n🏆 Best Performance:")
            best_speed = max(successful, key=lambda x: x.tokens_per_second)
            print(f"   Speed: {best_speed.tokens_per_second:.0f} tok/s")
            print(f"   Config: {best_speed.config.d_model}d, batch={best_speed.config.batch_size}, AMP={best_speed.config.use_amp}")
            
            best_memory = min(successful, key=lambda x: x.peak_memory_mb)
            print(f"\n💾 Best Memory:")
            print(f"   Memory: {best_memory.peak_memory_mb:.0f} MB")
            print(f"   Config: {best_memory.config.d_model}d, batch={best_memory.config.batch_size}, AMP={best_memory.config.use_amp}")

def main():
    """Main function to run experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AMP vs FP32 Experiment Runner")
    parser.add_argument("--test", action="store_true", help="Run in test mode (10 steps)")
    parser.add_argument("--full", action="store_true", help="Run in full mode (1000 steps)")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.test:
        test_mode = True
    elif args.full:
        test_mode = False
    else:
        # Default to test mode for safety
        test_mode = True
        print("🧪 Defaulting to TEST mode. Use --full for complete experiments.")
    
    # Run experiments
    runner = AMPExperimentRunner(test_mode=test_mode)
    results = runner.run_all_experiments()
    
    # Save and summarize
    runner.save_results()
    runner.print_summary()
    
    return results

if __name__ == "__main__":
    main()
