#!/usr/bin/env python3
"""
Simple script to summarize experiment results
"""
import json
import os
from pathlib import Path

results_dir = Path("results/ablation_batch_seqlen")

# Collect all results
all_results = []
for result_file in results_dir.glob("*_result.json"):
    with open(result_file) as f:
        data = json.load(f)
        all_results.append(data)

# Sort by validation loss
all_results.sort(key=lambda x: x['val_loss'])

print("\n" + "="*100)
print("EXPERIMENT 5 - MoE ABLATION RESULTS SUMMARY")
print("="*100)

print("\n📊 Results sorted by Validation Loss (best to worst):")
print("-"*100)
print(f"{'Rank':<6} {'Config Name':<25} {'Val Loss':<12} {'Val Acc':<12} {'Throughput':<16} {'Memory (GB)':<12}")
print("-"*100)

for i, result in enumerate(all_results, 1):
    name = result['config_name']
    val_loss = result['val_loss']
    val_acc = result['val_acc'] * 100
    throughput = result['throughput']
    memory = result.get('peak_memory_gb', 0)
    
    print(f"{i:<6} {name:<25} {val_loss:<12.4f} {val_acc:<11.2f}% {throughput:<15.0f} {memory:<12.2f}")

print("-"*100)

# Group by strategy
print("\n📈 Performance by Strategy:")
print("-"*100)

strategies = {
    'Large Batch (batch=104, seq=256)': [r for r in all_results if r['batch_size'] == 104 and r['seq_len'] == 256],
    'Long Sequence (batch=6, seq=4096)': [r for r in all_results if r['batch_size'] == 6 and r['seq_len'] == 4096],
    'Balanced (batch=26, seq=1024)': [r for r in all_results if r['batch_size'] == 26 and r['seq_len'] == 1024],
}

for strategy_name, results in strategies.items():
    if results:
        print(f"\n{strategy_name}:")
        for r in sorted(results, key=lambda x: x['val_loss']):
            print(f"  LR={r['lr']:6.4f}  →  Val Loss: {r['val_loss']:.4f}, Val Acc: {r['val_acc']*100:.2f}%, Throughput: {r['throughput']:.0f} tok/s")

# Find overall best
best = all_results[0]
print("\n" + "="*100)
print("🏆 BEST CONFIGURATION:")
print("="*100)
print(f"  Config: {best['config_name']}")
print(f"  Batch Size: {best['batch_size']}, Seq Length: {best['seq_len']}, Learning Rate: {best['lr']}")
print(f"  Val Loss: {best['val_loss']:.4f}")
print(f"  Val Accuracy: {best['val_acc']*100:.2f}%")
print(f"  Throughput: {best['throughput']:.0f} tokens/sec")
print(f"  Peak Memory: {best['peak_memory_gb']:.2f} GB")
print(f"  Training Time: {best['time']:.1f}s ({best['time']/60:.2f} min)")
print("="*100 + "\n")

# Learning rate analysis
print("\n📚 Learning Rate Analysis:")
print("-"*100)
lr_groups = {}
for r in all_results:
    lr = r['lr']
    if lr not in lr_groups:
        lr_groups[lr] = []
    lr_groups[lr].append(r)

for lr in sorted(lr_groups.keys()):
    results = lr_groups[lr]
    avg_loss = sum(r['val_loss'] for r in results) / len(results)
    avg_acc = sum(r['val_acc'] for r in results) / len(results)
    print(f"LR={lr:6.4f}:  Avg Val Loss: {avg_loss:.4f}, Avg Val Acc: {avg_acc*100:.2f}%  ({len(results)} configs)")

print("-"*100 + "\n")

