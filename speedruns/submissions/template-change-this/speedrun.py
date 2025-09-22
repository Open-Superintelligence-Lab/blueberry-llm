#!/usr/bin/env python3
"""
T4 Speedrun Challenge Entry Point
"""

import os
import sys
import time
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from speedrun.config import T4SpeedrunConfig, get_t4_speedrun_config, create_custom_t4_config
from data import load_and_cache_data, TextTokenDataset
from models import create_model
from training import train_model
from training.evaluation import evaluate_model


class SpeedrunTimer:
    """Timer for speedrun challenge"""
    
    def __init__(self, time_limit_minutes: int):
        self.time_limit_seconds = time_limit_minutes * 60
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def check_time_limit(self) -> bool:
        if self.start_time is None:
            return True
        elapsed = time.time() - self.start_time
        return elapsed < self.time_limit_seconds


def setup_data(config: T4SpeedrunConfig):
    """Setup data with fixed seed"""
    torch.manual_seed(config.SPEEDRUN_DATASET_SEED)
    
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    val_size = int(len(dataset) * config.SPEEDRUN_VAL_SPLIT)
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(config.SPEEDRUN_DATASET_SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, tokenizer


def run_speedrun(config: T4SpeedrunConfig, time_limit_minutes: int = None):
    """Run the T4 speedrun challenge"""
    
    if time_limit_minutes is not None:
        config.SPEEDRUN_TIME_LIMIT_MINUTES = time_limit_minutes
    
    timer = SpeedrunTimer(config.SPEEDRUN_TIME_LIMIT_MINUTES)
    
    print(f"T4 Speedrun: {config.SPEEDRUN_TIME_LIMIT_MINUTES} minutes")
    
    try:
        # Setup data
        train_loader, val_loader, tokenizer = setup_data(config)
        
        # Setup model
        model = create_model(config, "moe")
        model = model.cuda()
        
        # Start timer
        timer.start()
        
        # Train model
        model = train_model(model, train_loader, val_loader, config)
        
        # Final evaluation
        eval_metrics = evaluate_model(model, val_loader, config)
        
        # Results
        results = {
            'timestamp': datetime.now().isoformat(),
            'final_val_loss': eval_metrics['val_loss'],
            'total_time_minutes': timer.get_elapsed_time() / 60 if timer.start_time else 0,
            'completed': True
        }
        
        print(f"Final validation loss: {results['final_val_loss']:.6f}")
        print(f"Time: {results['total_time_minutes']:.2f} minutes")
        
        return results
        
    except Exception as e:
        print(f"Speedrun failed: {e}")
        return {'completed': False, 'error': str(e)}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="T4 Speedrun Challenge")
    parser.add_argument("--config", type=str, default="default", help="Configuration preset")
    parser.add_argument("--time-limit", type=int, default=5, help="Time limit in minutes")
    parser.add_argument("--d-model", type=int, help="Model dimension")
    parser.add_argument("--n-layers", type=int, help="Number of layers")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.d_model or args.n_layers or args.batch_size or args.lr:
        config_kwargs = {}
        if args.d_model: config_kwargs['d_model'] = args.d_model
        if args.n_layers: config_kwargs['n_layers'] = args.n_layers
        if args.batch_size: config_kwargs['batch_size'] = args.batch_size
        if args.lr: config_kwargs['muon_lr'] = args.lr
        config = create_custom_t4_config(**config_kwargs)
    else:
        config = get_t4_speedrun_config()
    
    # Run speedrun
    results = run_speedrun(config, args.time_limit)
    
    if not results['completed']:
        sys.exit(1)


if __name__ == "__main__":
    main()