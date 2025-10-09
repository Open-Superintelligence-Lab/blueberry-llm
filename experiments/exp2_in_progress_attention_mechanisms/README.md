# Attention Mechanisms Experiments

## Experiments

### exp1_sparse_vs_classic_attention
DeepSeek Sparse Attention vs classic multi-head attention

### exp2_mhla_sparse_comparison
MHLA with and without sparse attention

### experiment_3
Adaptive sparsity mechanisms

### exp4_lightning_indexer_optimization
Lightning Indexer optimization

### exp5_optimal_sparsity_analysis
Optimal sparsity level analysis

### exp6_qwen3_dsa_hybrid (NEW)
Compares 3 Qwen3-Next variants:
- Baseline: Standard Qwen3-Next
- DSA-Only: All DeepSeek Sparse Attention
- Hybrid: DSA for full attention, GatedDeltaNet for linear attention

```bash
cd exp6_qwen3_dsa_hybrid
python run_experiment.py  # Trains all 3 variants
```