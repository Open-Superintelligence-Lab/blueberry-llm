import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import json
from models import SimpleMoEModel, PreNormMoEBlock, PrePostNormMoEBlock
from config import ExperimentConfig
from data.loader import get_dataloaders


def train_model(model, train_loader, config, model_name):
    """Train a model and return loss history"""
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    losses = []
    model.train()
    
    print(f"\nTraining {model_name}...")
    for step, batch in enumerate(train_loader):
        if step >= config.num_steps:
            break
            
        input_ids = batch['input_ids'][:, :-1].to(config.device)
        targets = batch['input_ids'][:, 1:].to(config.device)
        
        logits, aux_loss = model(input_ids)
        
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), 
            targets.reshape(-1)
        )
        total_loss = loss + config.aux_loss_weight * aux_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % config.log_interval == 0:
            print(f"  Step {step+1}/{config.num_steps} | Loss: {loss.item():.4f} | Aux: {aux_loss.item():.4f}")
    
    return losses


def main():
    config = ExperimentConfig()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        num_workers=2
    )
    
    # Train Pre-Norm model
    print("\n" + "="*60)
    print("Experiment: Pre-Norm vs Pre+Post-Norm MoE Blocks")
    print("="*60)
    
    model_prenorm = SimpleMoEModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        num_experts=config.num_experts,
        top_k=config.top_k,
        dropout=config.dropout,
        use_pre_post_norm=False
    )
    
    losses_prenorm = train_model(model_prenorm, train_loader, config, "Pre-Norm Only")
    
    # Train Pre+Post-Norm model
    model_prepostnorm = SimpleMoEModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        num_experts=config.num_experts,
        top_k=config.top_k,
        dropout=config.dropout,
        use_pre_post_norm=True
    )
    
    losses_prepostnorm = train_model(model_prepostnorm, train_loader, config, "Pre+Post-Norm")
    
    # Save results
    results = {
        "config": vars(config),
        "prenorm_losses": losses_prenorm,
        "prepostnorm_losses": losses_prepostnorm,
        "prenorm_final": losses_prenorm[-1],
        "prepostnorm_final": losses_prepostnorm[-1],
        "prenorm_avg_last_50": sum(losses_prenorm[-50:]) / 50,
        "prepostnorm_avg_last_50": sum(losses_prepostnorm[-50:]) / 50
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Pre-Norm Final Loss:       {results['prenorm_final']:.4f}")
    print(f"Pre+Post-Norm Final Loss:  {results['prepostnorm_final']:.4f}")
    print(f"\nPre-Norm Avg (last 50):    {results['prenorm_avg_last_50']:.4f}")
    print(f"Pre+Post-Norm Avg (last 50): {results['prepostnorm_avg_last_50']:.4f}")
    print(f"\nImprovement: {((results['prenorm_avg_last_50'] - results['prepostnorm_avg_last_50']) / results['prenorm_avg_last_50'] * 100):.2f}%")
    print("\nResults saved to results/experiment_results.json")


if __name__ == "__main__":
    main()

